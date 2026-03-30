import itertools
# import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange, einsum, reduce, repeat

from posetail.posetail.cube import get_camera_scale
from posetail.posetail.cube import undistort_points, triangulate_simple_batch
from posetail.posetail.utils import PadToMultiple
from posetail.posetail.encoder_decoder import SceneRepresentation, QueryEncoder, Decoder

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class TrackerEncoder(nn.Module): 

    def __init__(self, track_3d = True, stride_length = 8, 
                 stride_overlap = None, downsample_factor = 4, 
                 hiera_requires_grad = False, vc_head_requires_grad = True,
                 cube_dim = 20, cube_extent = None, upsample_factor = 1, 
                 corr_levels = 4, corr_radius = 3, 
                 corr_hidden_dim = 384, corr_output_dim = 256, 
                 max_freq = 10, n_iters = 4, embedding_dim = 256, 
                 latent_dim = 128, n_virtual = 64, n_heads = 8, 
                 n_time_space_blocks = 6, embedding_factor = 4,
                 mode_3d = 'encoder'): 
        super().__init__()

        if track_3d: 
            self.R = 3 
        else: 
            self.R = 2

        self.mode_3d = mode_3d
            
        # video processing
        self.S = stride_length
        
        if stride_overlap is None: 
            self.stride_overlap = self.S // 2
        else:
            self.stride_overlap = stride_overlap 
        
        # cnn params
        self.downsample_factor = downsample_factor
        self.latent_dim = latent_dim 
        self.hiera_requires_grad = hiera_requires_grad

        # cube params
        self.cube_dim = cube_dim 
        self.cube_extent = cube_extent

        self.upsample_factor = upsample_factor

        # correlation params
        self.corr_levels = corr_levels 
        self.corr_radius = corr_radius 
        self.corr_dim = 2 * self.corr_radius + 1
        self.corr_hidden_dim = corr_hidden_dim
        self.corr_output_dim = corr_output_dim

        # transformer params
        self.max_freq = max_freq     
        self.n_iters = n_iters
        self.n_virtual = n_virtual
        self.embedding_dim = embedding_dim   
        self.n_heads = n_heads
        self.n_time_space_blocks = n_time_space_blocks
        self.embedding_factor = embedding_factor
        self.activation_kwargs = {'approximate': 'tanh'}
        self.vc_head_requires_grad = vc_head_requires_grad

        self.n_frames = 16
        
        # self.transform_norm = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.transform_norm = transforms.Compose([
            PadToMultiple(16),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])


        self.scene_encoder = SceneRepresentation(
            version='large',
            freeze_encoder=hiera_requires_grad
        )
        
        self.query_encoder = QueryEncoder(
            embed_dim=embedding_dim,
            decoder_dim=latent_dim,
            n_frames=self.n_frames, 
            vdim=8, 
            corr_radius=corr_radius, 
            max_freq=self.max_freq
        )
        self.decoder = Decoder(
            embed_dim=latent_dim,
            encoder_dim=self.scene_encoder.embed_dim,
            num_heads=n_heads,
            num_layers=n_time_space_blocks,
            mlp_ratio=embedding_factor
        )

    def forward(self, views, coords, camera_group = None):
        '''
        B: batch size
        T: number of frames in video
        C: number of channels 
        H: height of image
        W: width of image
        D: latent dimension
        '''
        device = coords.device

        B, N, R = coords.shape
        B, T, H, W, C = views[0].shape

        assert R == self.R
        assert self.n_frames == T

        if self.R == 3:
            self.cube_scale = get_camera_scale(camera_group, coords.reshape(-1, 3))

        # normalize frames
        views_norm = []
        for i, frames in enumerate(views): 
            # frames = 2 * (frames / 255.0) - 1
            frames = frames.to(device)
            frames = rearrange(frames, 'b t h w c -> b t c h w')
            frames = self.transform_norm(frames)
            views_norm.append(frame)

        scene_features = self.scene_encoder(views_norm)

        # have coords at 0
        query_coords = repeat(coords, 'b n r -> b (t n) r', t=T)
        query_time = torch.zeros((B, T * N), dtype=torch.int32, device=device)
        target_time = repeat(torch.arange(T, device='cuda'), 't -> b (t n)', b=B, t=T, n=N)
        
        query_embeds, visible = self.query_encoder(
            views_norm, camera_group,
            query_coords = query_coords,
            query_time = query_time,
            target_time = target_time,
            cube_scale = self.cube_scale
        )

        outputs = self.decoder(scene_features, query_embeds, visible)
        outputs = rearrange(outputs, 'b (t n) cams outdim -> cams b t n outdim',
                            t=T, n=N)

        points_pred, vis_pred, conf_pred, depth_pred = torch.split(
            outputs, [2, 1, 1, 1], dim=-1
        )

        depth_pred_scaled = (depth_pred + 0.5) * cube_scale * 500
        
        sizes = torch.stack([cam['size'] for cam in camera_group])
        points_pred_scaled = points_pred * sizes
        
        points_und = torch.stack([
            undistort_points(camera_group[i], points_pred_scaled[i])
            for i in range(len(camera_group))
        ]) 

        points_und_flat = rearrange(points_und, 'cams b t n r -> cams (b t n) r')
        camera_mats = torch.stack([cam['ext'] for cam in camera_group])
        weights = rearrange(conf_pred, 'cams b t n -> cams (b t n)')
        points_3d_flat = triangulate_simple_batch(points_und_flat, camera_mats, weights)
        points_3d = rearrange(points_3d, '(b t n) r -> b t n r', b=B, t=T, n=N)
        
        # assemble outputs 
        result_dict = {
            'coords_pred': points_3d, # (b, t, n, 3)
            '2d_pred': points_pred_scaled, # (cams, b, t, n, 2)
            'vis_pred': vis_pred, # (cams, b, t, n)
            'conf_pred': conf_pred, # (cams, b, t, n)
            'depth_pred': depth_pred_scaled # (cams, b, t, n)
        }

        # if self.training: 
        #     train_dict = {
        #         'coords_pred_iters': coords_pred_iters,
        #         'vis_pred_iters': vis_pred_iters, 
        #         'conf_pred_iters': conf_pred_iters}
            
        #     result_dict.update(train_dict)

        return result_dict 
