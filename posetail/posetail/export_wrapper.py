"""
TrackerEncoderONNX — export-friendly wrapper around TrackerEncoder.

Interface differences vs the original forward:
  • views is a single stacked float tensor [cams, B, T, H, W, C] instead of a
    Python list of tensors.
  • camera_group is replaced by 7 stacked tensors: cam_ext, cam_ext_inv,
    cam_center, cam_mat, cam_dist, cam_size, cam_offset.
  • cube_scale is computed inside the wrapper using the export-safe
    get_camera_scale_export (no .item() call).
  • Returns a flat tuple of tensors rather than a Python dict.

Dynamic axes (declared in export_onnx.py):
  • cams: number of cameras
  • B:    batch size
  • N:    number of query points

Fixed at export time (256 × 256, 16 frames):
  • T = 16, H = 256, W = 256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, einsum as esum

from posetail.posetail.cube import to_homogeneous, from_homogeneous, _invert_SE3
from posetail.posetail.cube_batched import (
    project_cam_batched, undistort_points_batched,
    points_to_rays_batched, get_camera_scale_export,
)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _mat3x3_inv(A):
    """Analytical 3×3 batch inverse via cofactors.  A: [..., 3, 3] → [..., 3, 3]."""
    a = A[..., 0, 0]; b = A[..., 0, 1]; c = A[..., 0, 2]
    d = A[..., 1, 0]; e = A[..., 1, 1]; f = A[..., 1, 2]
    g = A[..., 2, 0]; h = A[..., 2, 1]; i = A[..., 2, 2]

    C00 =  (e*i - f*h);  C01 = -(d*i - f*g);  C02 =  (d*h - e*g)
    C10 = -(b*i - c*h);  C11 =  (a*i - c*g);  C12 = -(a*h - b*g)
    C20 =  (b*f - c*e);  C21 = -(a*f - c*d);  C22 =  (a*e - b*d)

    det = a*C00 + b*C01 + c*C02
    inv_det = 1.0 / det.abs().clamp(min=1e-10) * det.sign().clamp(min=1.0)

    adj = torch.stack([
        torch.stack([C00, C10, C20], dim=-1),
        torch.stack([C01, C11, C21], dim=-1),
        torch.stack([C02, C12, C22], dim=-1),
    ], dim=-2)
    return adj * rearrange(inv_det, '... -> ... 1 1')


class TrackerEncoderONNX(nn.Module):
    """ONNX-exportable wrapper around a TrackerEncoder instance."""

    def __init__(self, base_model):
        super().__init__()
        self.m = base_model

    def forward(
        self,
        views,        # [cams, B, T, H, W, C]  float32
        coords,       # [B, N, 3]               float32
        query_times,  # [B, N]                  int64
        cam_ext,      # [cams, 4, 4]
        cam_mat,      # [cams, 3, 3]
        cam_dist,     # [cams, 5]
        cam_offset,   # [cams, 2]
    ):
        """
        Returns a 13-tuple of tensors (same content as TrackerEncoder's result_dict):
            coords_pred          [B, T, N, 3]
            3d_pred_cams_direct  [cams, B, T, N, 3]
            3d_pred_cams_rays    [cams, B, T, N, 3]
            conf_3d              [cams, B, T, N]
            3d_pred_direct       [B, T, N, 3]
            3d_pred_rays         [B, T, N, 3]
            3d_pred_triangulate  [B, T, N, 3]
            2d_pred              [cams, B, T, N, 2]
            vis_pred             [B, T, N, 1]
            conf_pred            [B, T, N, 1]
            vis_pred_2d          [cams, B, T, N]
            conf_pred_2d         [cams, B, T, N]
            depth_pred           [cams, B, T, N]
        """
        cams_n = views.shape[0]
        B = views.shape[1]
        T = views.shape[2]
        N = coords.shape[1]

        cam_ext_inv = _invert_SE3(cam_ext)             # [cams, 4, 4]
        cam_center  = cam_ext_inv[:, :3, 3]            # translation col = -R^T t  [cams, 3]
        cam_size    = torch.tensor([[256.0, 256.0]], device=views.device,
                                   dtype=torch.float32).expand(cams_n, -1).contiguous()

        # ── Normalise views ──────────────────────────────────────────────────
        # [cams, B, T, H, W, C] -> [cams, B, T, C, H, W]
        views_chw = rearrange(views.float(), 'c b t h w ch -> c b t ch h w')
        mean = torch.tensor(_IMAGENET_MEAN, device=views.device, dtype=torch.float32)
        std  = torch.tensor(_IMAGENET_STD,  device=views.device, dtype=torch.float32)
        mean_b = rearrange(mean, 'ch -> 1 1 1 ch 1 1')
        std_b  = rearrange(std,  'ch -> 1 1 1 ch 1 1')
        views_norm = (views_chw - mean_b) / std_b   # [cams, B, T, C, H, W]

        # ── Cube scale ───────────────────────────────────────────────────────
        cube_scale = get_camera_scale_export(
            cam_ext, cam_mat, cam_dist, cam_offset, cam_size,
            rearrange(coords.float(), 'b n r -> (b n) r'))

        # ── Scene encoding ───────────────────────────────────────────────────
        enc = self.m.scene_encoder
        if enc.freeze_encoder:
            enc.encoder.eval()
        # [cams, B, T, C, H, W] -> [(cams*B), C, T, H, W]
        xr = rearrange(views_norm, 'c b t ch h w -> (c b) ch t h w')
        with torch.set_grad_enabled(not enc.freeze_encoder):
            feat = enc.encoder(xr)           # [(cams*B), n_tokens, embed_dim]
        feat = feat + enc.pos_embed
        scene_features = rearrange(feat, '(c b) tokens embed -> c b tokens embed',
                                   c=cams_n, b=B)

        # ── Query coordinates ────────────────────────────────────────────────
        query_coords    = repeat(coords, 'b n r -> b (t n) r', t=T).float()
        query_times_rep = repeat(query_times.long(), 'b n -> b (t n)', t=T)
        target_time     = repeat(torch.arange(T, device=coords.device, dtype=torch.long),
                                 't -> b (t n)', b=B, t=T, n=N)

        # ── Query encoding (batched) ─────────────────────────────────────────
        query_embeds, _ = self.m.query_encoder.forward_batched(
            views_norm,
            cam_ext, cam_mat, cam_dist, cam_offset, cam_size, cam_center,
            query_coords, query_times_rep, target_time, cube_scale,
        )   # [B, T*N, cams, decoder_dim]

        # ── 2-D query positions ──────────────────────────────────────────────
        p2d_query_tn = project_cam_batched(cam_ext, cam_mat, cam_dist, cam_offset, query_coords)
        # [cams, B, T*N, 2]
        p2d_query = rearrange(p2d_query_tn, 'c b (t n) r -> c b t n r', t=T, n=N)

        # ── Ray matrices ─────────────────────────────────────────────────────
        p2d_flat = rearrange(p2d_query, 'c b t n r -> c (b t n) r')
        query_rays_flat = points_to_rays_batched(
            cam_ext, cam_mat, cam_dist, cam_offset, p2d_flat, cube_scale)
        # [cams, B*T*N, 4, 4]
        query_rays = rearrange(query_rays_flat, 'c (b t n) d e -> b (t n) c d e',
                               b=B, t=T, n=N)

        # ── Decoder ──────────────────────────────────────────────────────────
        outputs = self.m.decoder(scene_features, query_embeds, query_rays)
        outputs = rearrange(outputs, 'b (t n) c outdim -> c b t n outdim', t=T, n=N)

        (points_3d_offsets, points_pred, vis_pred_2d_logits,
         conf_pred_2d_logits, depth_pred_raw, conf_3d_logits) = torch.split(
            outputs, [3, 2, 1, 1, 1, 1], dim=-1)

        vis_pred_2d  = torch.sigmoid(vis_pred_2d_logits)
        conf_pred_2d = torch.sigmoid(conf_pred_2d_logits)
        conf_3d      = torch.softmax(conf_3d_logits[..., 0], dim=0)  # [cams, B, T, N]

        # ── Depth prediction ─────────────────────────────────────────────────
        qc_tn   = rearrange(query_coords, 'b (t n) r -> b t n r', t=T, n=N)
        qc_exp  = rearrange(qc_tn, 'b t n r -> b t n 1 r')
        centers = cam_center.to(coords.dtype)   # [cams, 3]
        depths_query = torch.linalg.norm(qc_exp - centers, dim=-1)   # [B, T, N, cams]
        depths_query_shaped = rearrange(depths_query, 'b t n c -> c b t n')
        depth_pred_scaled   = depths_query_shaped + depth_pred_raw[..., 0] * cube_scale * self.m.depth_scale

        # ── 2-D offset predictions ────────────────────────────────────────────
        points_pred_scaled = p2d_query + points_pred * self.m.p2d_scale

        # ── 3-D (direct via camera-space offsets + unproject) ────────────────
        qc_flat = query_coords  # [B, T*N, 3]
        query_coords_cams_flat = from_homogeneous(
            esum(to_homogeneous(qc_flat), cam_ext, 'b tn r, c x r -> c b tn x'))
        # [cams, B, T*N, 3]
        query_coords_cams = rearrange(query_coords_cams_flat, 'c b (t n) r -> c b t n r', t=T, n=N)

        p3d_cams     = query_coords_cams + points_3d_offsets * cube_scale * self.m.p3d_scale
        p3d_cams_hom = to_homogeneous(p3d_cams)  # [cams, B, T, N, 4]
        points_3d_all_direct = from_homogeneous(
            esum(p3d_cams_hom.float(), cam_ext_inv.float(), 'c b t n r, c x r -> c b t n x')
        ).to(p3d_cams.dtype)   # [cams, B, T, N, 3]

        points_3d_direct = esum(points_3d_all_direct, conf_3d, 'c b t n r, c b t n -> b t n r')

        # ── 3-D (ray-based) ──────────────────────────────────────────────────
        points_und = undistort_points_batched(cam_mat, cam_dist, cam_offset,
                                              points_pred_scaled)   # [cams, B, T, N, 2]
        rays_norm  = to_homogeneous(points_und)                     # [cams, B, T, N, 3]
        rot_mats   = cam_ext[:, :3, :3]                             # [cams, 3, 3]
        rays_world = esum(rays_norm.float(), rot_mats.float(), 'c b t n r, c r x -> c b t n x')
        rays_world = F.normalize(rays_world.to(rays_norm.dtype), dim=-1)

        cadd = rearrange(centers, 'c r -> c 1 1 1 r')
        points_3d_all_rays = cadd + rays_world * rearrange(depth_pred_scaled, 'c b t n -> c b t n 1')
        points_3d_rays = esum(points_3d_all_rays, conf_3d, 'c b t n r, c b t n -> b t n r')

        # ── Triangulation (midpoint, SVD-free) ───────────────────────────────
        # Uses already-computed rays_world [cams, B, T, N, 3] and centers [cams, 3].
        # Solves:  sum_c w_c (I - d_c d_c^T) P = sum_c w_c (I - d_c d_c^T) o_c
        w_tri = conf_pred_2d[..., 0]           # [cams, B, T, N]
        ddt   = esum(rays_world, rays_world,
                     'c b t n i, c b t n j -> c b t n i j')   # [cams, B, T, N, 3, 3]
        I3    = torch.eye(3, device=rays_world.device, dtype=rays_world.dtype)
        M_mat = I3 - ddt                       # [cams, B, T, N, 3, 3]
        w_exp = rearrange(w_tri, 'c b t n -> c b t n 1 1')
        A_tri = (w_exp * M_mat).sum(dim=0)     # [B, T, N, 3, 3]
        Mo    = esum(M_mat, centers.to(rays_world.dtype),
                     'c b t n i j, c j -> c b t n i')  # [cams, B, T, N, 3]
        b_tri = (rearrange(w_tri, 'c b t n -> c b t n 1') * Mo).sum(dim=0)  # [B, T, N, 3]
        A_flat = rearrange(A_tri, 'b t n i j -> (b t n) i j')
        b_flat = rearrange(b_tri, 'b t n i   -> (b t n) i')
        p_flat = esum(_mat3x3_inv(A_flat), b_flat, 'm i j, m j -> m i')  # [(b t n), 3]
        points_3d_tri = rearrange(p_flat, '(b t n) r -> b t n r', b=B, t=T, n=N)

        # ── Aggregate visibility / confidence ─────────────────────────────────
        vis_pred  = torch.amax(vis_pred_2d,  dim=0)   # [B, T, N, 1]
        conf_pred = torch.amax(conf_pred_2d, dim=0)   # [B, T, N, 1]

        return (
            points_3d_direct,                    # coords_pred          [B, T, N, 3]
            points_3d_all_direct,                # 3d_pred_cams_direct  [cams, B, T, N, 3]
            points_3d_all_rays,                  # 3d_pred_cams_rays    [cams, B, T, N, 3]
            conf_3d,                             # conf_3d              [cams, B, T, N]
            points_3d_direct,                    # 3d_pred_direct       [B, T, N, 3]
            points_3d_rays,                      # 3d_pred_rays         [B, T, N, 3]
            points_3d_tri,                       # 3d_pred_triangulate  [B, T, N, 3]
            points_pred_scaled,                  # 2d_pred              [cams, B, T, N, 2]
            vis_pred,                            # vis_pred             [B, T, N, 1]
            conf_pred,                           # conf_pred            [B, T, N, 1]
            vis_pred_2d_logits[..., 0],          # vis_pred_2d          [cams, B, T, N]
            conf_pred_2d_logits[..., 0],         # conf_pred_2d         [cams, B, T, N]
            depth_pred_scaled,                   # depth_pred           [cams, B, T, N]
        )
