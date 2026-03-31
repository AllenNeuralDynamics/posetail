import itertools
# import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange, einsum, reduce, repeat

from posetail.posetail.cube import get_camera_scale, from_homogeneous, to_homogeneous
from posetail.posetail.cube import undistort_points, triangulate_simple_batch, project_points_torch
from posetail.posetail.utils import PadToMultiple, PadToSize
from posetail.posetail.encoder_decoder import SceneRepresentation, QueryEncoder, Decoder

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

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
            PadToSize(256),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])


        self.scene_encoder = SceneRepresentation(
            version='giant',
            freeze_encoder=~hiera_requires_grad
        )
        
        self.query_encoder = QueryEncoder(
            embed_dim=embedding_dim,
            decoder_dim=latent_dim,
            n_frames=self.n_frames, 
            vdim=16, 
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

        self.p2d_scale = nn.Parameter(torch.tensor([32.0]))

        self.depth_scale = nn.Parameter(torch.tensor([500.0]))


    def print_summary(self):
        print("PARAMETERS")
        print("  total parameters: {:,d}".format(count_parameters(self)))
        print("  query encoder params: {:,d}".format(count_parameters(self.query_encoder)))
        print("  scene representation params: {:,d}".format(count_parameters(self.scene_encoder)))
        print("  decoder params: {:,d}".format(count_parameters(self.decoder)))
        
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
            views_norm.append(frames)

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
        outputs = rearrange(outputs,
                            'b (t n) cams outdim -> cams b t n outdim',
                            t=T, n=N)

        points_pred, vis_pred_2d, conf_pred_2d_raw, depth_pred = torch.split(
            outputs, [2, 1, 1, 1], dim=-1
        )
        
        depth_pred_scaled = (depth_pred[..., 0] + 1) * self.cube_scale * self.depth_scale

        vis_pred_2d = F.sigmoid(vis_pred_2d)
        conf_pred_2d = F.sigmoid(conf_pred_2d_raw)

        # zero out confidences for points outside of range
        # bad = ( (points_pred[..., 0] < -1.5) | (points_pred[..., 0] > 1.5) |
        #         (points_pred[..., 1] < -1.5) | (points_pred[..., 1] > 1.5) ) 
        # conf_pred_2d = einsum(conf_pred_2d, ~bad, 'cams b t n r, cams b t n -> cams b t n r')

        # clip points and get scales
        # sizes = torch.stack([cam['size'] for cam in camera_group])
        # points_pred = torch.clip(points_pred, -1.5, 1.5)
        # points_pred_scaled = einsum((points_pred + 1) * 0.5, sizes,
        #                             'cams b t n r, cams r -> cams b t n r')
        # points_pred_scaled = F.sigmoid(points_pred) * 256
        # points_pred_scaled = (points_pred + 1) * self.p2d_scale
        
        p2d_query = project_points_torch(camera_group, query_coords) # [cams, b, (t n), 2]
        p2d_query = rearrange(p2d_query, 'cams b (t n) r -> cams b t n r', t=T)
        # Predict offsets instead of absolute bounded coordinates
        points_pred_scaled = p2d_query + points_pred * self.p2d_scale 
        
        points_und = torch.stack([
            undistort_points(camera_group[i], points_pred_scaled[i])
            for i in range(len(camera_group))
        ])

        # get 3d points from each cameras using rays
        rays_norm = to_homogeneous(points_und)
        rot_mats = torch.stack([cam['ext'][:3,:3] for cam in camera_group])
        rays_world = einsum(rays_norm, rot_mats, 'cams b t n r, cams r x -> cams b t n x')
        rays_world = F.normalize(rays_world, dim=-1)

        centers = torch.stack([cam['center'] for cam in camera_group])
        cadd = repeat(centers, 'cams r -> cams 1 1 1 r')
        points_3d_all = cadd + einsum(rays_world, depth_pred_scaled,
                                      'cams b t n r, cams b t n -> cams b t n r')

        # # average 3d predictions
        conf_3d = torch.softmax(conf_pred_2d_raw[..., 0], dim=0)
        points_3d = einsum(points_3d_all, conf_3d, 'cams b t n r, cams b t n -> b t n r')


        # triangulate points
        # points_und_flat = rearrange(points_und, 'cams b t n r -> cams (b t n) r')
        # camera_mats = torch.stack([cam['ext'] for cam in camera_group])
        # weights = rearrange(conf_pred_2d, 'cams b t n 1 -> cams (b t n)')
        # points_und_flat = torch.clip(points_und_flat, -2, 2)
        # points_3d_flat = triangulate_simple_batch(points_und_flat, camera_mats, weights)
        # points_3d = rearrange(points_3d_flat, '(b t n) r -> b t n r', b=B, t=T, n=N)

        # # zero out 3d points with no confidence
        # bad_pred = torch.amax(conf_pred_2d[..., 0], dim=0) <= 1e-5
        # points_3d = einsum(points_3d, ~bad_pred, 'b t n r, b t n -> b t n r') 
        
        vis_pred = torch.amax(vis_pred_2d, dim=0)
        conf_pred = torch.amax(conf_pred_2d, dim=0)
        
        # assemble outputs 
        result_dict = {
            'coords_pred': points_3d, # (b, t, n, 3)
            '3d_pred_cams': points_3d_all, # (cams, b, t, n, 3)
            '2d_pred': points_pred_scaled, # (cams, b, t, n, 2)
            'vis_pred': vis_pred, # (b, t, n, 1)
            'conf_pred': conf_pred, # (b, t, n, 1)
            'vis_pred_2d': vis_pred_2d[..., 0], # (cams, b, t, n)
            'conf_pred_2d': conf_pred_2d[..., 0], # (cams, b, t, n)
            'depth_pred': depth_pred_scaled # (cams, b, t, n)
        }

        # if self.training: 
        #     train_dict = {
        #         'coords_pred_iters': coords_pred_iters,
        #         'vis_pred_iters': vis_pred_iters, 
        #         'conf_pred_iters': conf_pred_iters}
            
        #     result_dict.update(train_dict)

        return result_dict 
