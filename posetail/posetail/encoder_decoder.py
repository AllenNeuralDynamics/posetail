#!/usr/bin/env python3

import torch
import torch.nn as nn 
import torch.nn.functional as F

from posetail.posetail.networks import EmbedV2V
from posetail.posetail.cube import is_point_visible, project_points_torch

from einops import rearrange

def sample_feature_cubes_time(feature_planes, camera_group,
                              cube_centers, query_time, cube_interval,
                              corr_radius=1, downsample_ratio=1,
                              v2v=None):
    """Inputs:
     feature_planes: list of [B, T, C, H, W] tensors (one per camera)
     camera_group: list of cameras
     cube_centers: b k 3
     query_time: b k (time index for each query)
     cube_interval: single float
    
    Returns:
      volume: b d k total
    """
        
    cube_size = corr_radius * 2 + 1
    n_cams = len(feature_planes)
    B, K, _ = cube_centers.shape
    
    # get coordinates of each cube
    row = (torch.arange(cube_size) - corr_radius) * cube_interval
    xyz_s = torch.stack(torch.meshgrid(row, row, row, indexing='ij'))
    xyz = rearrange(xyz_s, 'r x y z -> (x y z) r')
    xyz = xyz.contiguous().to(device=cube_centers.device, dtype=cube_centers.dtype)

    cube_coords = cube_centers[..., None, :] + xyz
    cube_coords_flat = rearrange(cube_coords, 'b k total r -> (b k total) r')
    p2d_flat = project_points_torch(
        camera_group=camera_group, 
        coords_3d=cube_coords_flat, 
        downsample_factor=downsample_ratio,
    )

    p2d = rearrange(p2d_flat, 'ncams (b k total) r -> ncams b k total r',
                    b=B, k=K)

    all_samples = []
    all_masks = []
    for ix_cam in range(n_cams):
        b, t, d, h, w = feature_planes[ix_cam].shape
        scale = torch.tensor([w, h], device=p2d.device) 
        p2d_scaled = 2 * p2d[ix_cam] / scale - 1

        # Create visibility mask: True if within [-1, 1] bounds
        valid_mask = ((p2d_scaled[..., 0] >= -1) & (p2d_scaled[..., 0] <= 1) &
                      (p2d_scaled[..., 1] >= -1) & (p2d_scaled[..., 1] <= 1))
        
        # Gather the relevant time slices for each query: [B, K, C, H, W]
        b_idx = torch.arange(b, device=feature_planes[ix_cam].device)[:, None].expand(-1, K)
        feats_gathered = feature_planes[ix_cam][b_idx, query_time]  # b k d h w
        
        # For grid_sample, we need [B*K, C, H, W] input and [B*K, total, 1, 2] grid
        feats_flat = rearrange(feats_gathered, 'b k d h w -> (b k) d h w')
        grid_flat = rearrange(p2d_scaled, 'b k total r -> (b k) total 1 r')
        
        samples_flat = F.grid_sample(
            input=feats_flat,
            grid=grid_flat,
            align_corners=False,
            padding_mode="zeros")  # (b k) d total 1
        
        samples = rearrange(samples_flat[..., 0], '(b k) d total -> b d k total', b=b, k=K)
        all_samples.append(samples)
        all_masks.append(valid_mask)

    # volumes: cams b d k total            
    volumes = torch.stack(all_samples)
    masks = torch.stack(all_masks)  # ncams b k total

    masks_expanded = repeat(masks, "ncams b k total -> ncams b d k total",
                            d=volumes.shape[2])

    # apply softmax from learnable triangulation
    volumes[~masks_expanded] = -1e3
    weights = F.softmax(volumes, dim=0)
    mean_volume = torch.sum(volumes * weights, dim=0)

    mv_flat = rearrange(mean_volume, 'b d k (x y z) -> (b k) d z y x',
                        x=cube_size, y=cube_size, z=cube_size)
    if v2v is not None:
        mv_flat = v2v(mv_flat)

    mv_flat = F.normalize(mv_flat, p=2, dim=1, eps=1e-6)
    
    mean_volume = rearrange(mv_flat, '(b k) d z y x -> b d k (x y z)',
                            b=B, k=K) 

    return mean_volume



class QueryEncoder(nn.Module):
    def __init__(self, embed_dim=256, n_frames=16, vdim=8, corr_radius=2, max_freq=10, feature_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim
        self.corr_radius = corr_radius
        self.max_freq = max_freq
        
        # Time embeddings
        self.t_query_embed = nn.Embedding(n_frames, embed_dim)
        self.t_target_embed = nn.Embedding(n_frames, embed_dim)
        
        # Volume processing
        self.v2v = EmbedV2V(3, vdim)
        in_dim_vol = (corr_radius * 2 + 1) ** 3 * vdim
        self.linear_volume = nn.Linear(in_dim_vol, embed_dim)
        
        # Positional encodings
        self.linear_pos = nn.Linear(4 * max_freq, embed_dim)
        self.linear_depth = nn.Linear(2 * max_freq, embed_dim)
        
    def forward(self, preprocessed_views, camera_group,
                query_coords, query_time, target_time,
                cube_scale):
        """
        Args:
            preprocessed_views: list of [B, T, C, H, W] tensors (normalized features)
            camera_group: list of camera dicts 
            query_coords: [B, T_query, 3] 3D positions
            query_time: [B, T_query] time indices
            target_time: [B, T_query] time indices  
            cube_scale: float for cube sampling
            
        Returns:
            embed_total: [B, T_query, N_cams, embed_dim] embeddings per camera
            visible: [B, T_query, N_cams] which points are visible
        """
        B, T_query, _ = query_coords.shape
        
        # Sample feature volumes
        volumes = sample_feature_cubes_time(
            preprocessed_views, camera_group, query_coords, query_time,
            cube_scale * 2, corr_radius=self.corr_radius, v2v=self.v2v)
        volumes = rearrange(volumes, 'b d t total -> b t 1 (d total)')
        
        # Embed volume features
        embed_patch = self.linear_volume(volumes)
        
        # Time embeddings
        embed_query_time = rearrange(self.t_query_embed(query_time),
                                     'b t embed -> b t 1 embed')
        embed_target_time = rearrange(self.t_target_embed(target_time),
                                      'b t embed -> b t 1 embed')
        
        # 2D position encoding
        sizes = torch.stack([cam['size'] for cam in camera_group])
        p2d_full = project_points_torch(camera_group, query_coords)
        pp = rearrange(p2d_full, 'ncams b t r -> b t ncams r', b=B) / sizes
        fourier_pos = get_fourier_encoding(pp, min_freq=0, max_freq=self.max_freq)
        embed_pos = self.linear_pos(fourier_pos)

        qflat = rearrange(query_coords, 'b t r -> (b t) r')
        visible = []
        for cam in camera_group:
            out = is_point_visible(cam, qflat, margin=2)
            visible.append(out)
        visible = torch.stack(visible)
        visible = rearrange(visible, 'ncams (b t) -> b t ncams', b=B)
        
        # Depth encoding
        centers = torch.stack([cam['center'] for cam in camera_group])
        qc = rearrange(query_coords, 'b t r -> b t 1 r')
        depths = torch.linalg.norm(qc - centers, dim=-1) / (cube_scale * 500) - 0.5
        dr = rearrange(depths, "b t ncams -> b t ncams 1", b=B)
        fourier_depth = get_fourier_encoding(dr, min_freq=0, max_freq=self.max_freq)
        embed_depth = self.linear_depth(fourier_depth)
        
        # Combine all embeddings
        embed_total = embed_patch + embed_query_time + \
            embed_target_time + embed_pos + embed_depth

        return embed_total, visible

class SceneRepresentation(nn.Module):
    def __init__(self, version='large', freeze_encoder=True):
        super().__init__()
        
        # Initialize encoder
        if version == 'base':
            vjepa_encoder, vjepa_decoder = vjepa2_1_vit_base_384()
        elif version == 'large':
            vjepa_encoder, vjepa_decoder = vjepa2_1_vit_large_384()
        elif version == 'giant':
            vjepa_encoder, vjepa_decoder = vjepa2_1_vit_giant_384()
        elif version == 'gigantic':
            vjepa_encoder, vjepa_decoder = vjepa2_1_vit_gigantic_384()
        
        self.encoder = vjepa_encoder

        self.embed_dim = self.encoder.embed_dim
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
        self.freeze_encoder = freeze_encoder
        
    def forward(self, views):
        """
        Args:
            views: list of [B, T, H, W, C] tensors (preprocessed images, one list per camera)
            
        Returns:
            encoded_views: list of [B, N_tokens, encoder_dim] tensors (one per camera)
        """
        encoded = []
        
        # Set encoder to eval mode if frozen
        if self.freeze_encoder:
            self.encoder.eval()
        
        for imgs in views:
            xr = rearrange(imgs, 'b t c h w -> b c t h w')
            
            # Encode
            with torch.set_grad_enabled(not self.freeze_encoder):
                feat = self.encoder(xr)
            
            encoded.append(feat)
        
        return encoded
    


class QueryDecoder(nn.Module):
    def __init__(self, embed_dim=256, encoder_dim=1024,
                 num_heads=8, num_layers=8, 
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.num_layers = num_layers
        
        # Cross-attention layers
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                kdim=encoder_dim,
                vdim=encoder_dim,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # MLP layers (2-layer MLP with expansion)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norm1s = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.norm2s = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # Final output head
        self.output_head = nn.Linear(embed_dim, 2 + 1 + 1 + 1)  # 2d, depth, conf, vis
        
    def forward(self, scene_features, query_embeds, visible):
        """
        Args:
            scene_features: list of [B, N_tokens, encoder_dim] from SceneRepresentation
            query_embeds: [B, T_query, N_cams, embed_dim] from QueryEncoder
            visible: [B, T_query, N_cams] which ones to use
        Returns:
            outputs: [B, T_query, N_cams, 5] predictions (2d pos, depth, conf, vis)
        """
        B, T_query, N_cams, _ = query_embeds.shape
        
        # Process each camera separately
        outputs_per_cam = []
        
        for cam_idx in range(N_cams):
            # Get query for this camera: [B, T_query, embed_dim]
            query = query_embeds[:, :, cam_idx, :]
            
            # Get scene features for this camera: [B, N_tokens, encoder_dim]
            kv = scene_features[cam_idx]

            # Get visibility mask for this camera: [B, T_query]
            vis_mask = visible[:, :, cam_idx]
            
            # Apply cross-attention + MLP layers
            x = query
            for layer_idx in range(self.num_layers):
                # Cross-attention with residual
                attn_out, _ = self.cross_attns[layer_idx](
                    query=x,
                    key=kv,
                    value=kv,
                    need_weights=False
                )
                x = self.norm1s[layer_idx](attn_out)
                
                # MLP with residual
                mlp_out = self.mlps[layer_idx](x)
                x = self.norm2s[layer_idx](x + mlp_out)

            # Project to output: [B, T_query, 5]
            output = self.output_head(x)

            # Mask out invisible predictions
            # Shape: [B, T_query, 1] to broadcast across the 5 output dims
            vis_mask_expanded = rearrange(vis_mask, 'b t -> b t 1')
            output = output * vis_mask_expanded.float()
            
            outputs_per_cam.append(output)
            
                    
        # Stack outputs: [B, T_query, N_cams, 5]
        outputs = torch.stack(outputs_per_cam, dim=2)
        
        return outputs
    



