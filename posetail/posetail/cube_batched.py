"""
Batched (stacked-tensor) camera-space operations for ONNX export.

Camera parameters are stacked along a leading `cams` dimension rather than
stored in a Python list of dicts, so no Python loops over cameras remain in
the forward graph.
"""

import torch
import torch.nn.functional as F

from einops import rearrange, einsum as esum

from posetail.posetail.cube import to_homogeneous, from_homogeneous


def project_cam_batched(ext, mat, dist, offset, p3d, max_normalized=3.0):
    """Project 3D points through all cameras simultaneously.

    Args:
        ext:    [cams, 4, 4]  world-to-camera extrinsics
        mat:    [cams, 3, 3]  camera intrinsics
        dist:   [cams, D]     distortion coefficients (D >= 5)
        offset: [cams, 2]     pixel offset subtracted after projection
        p3d:    [..., 3]      3D points (any leading dims)

    Returns:
        p2d: [cams, ..., 2]
    """
    orig_shape = p3d.shape          # e.g. [B, N, 3] or [M, 3]
    p3d_flat = rearrange(p3d, '... r -> (...) r')  # [M, 3]

    # Project to camera space: [cams, M, 4]
    p_cam = esum(to_homogeneous(p3d_flat), ext, 'm r, c x r -> c m x')
    p2d_raw = from_homogeneous(p_cam[..., :3])     # [cams, M, 2]
    p2d_raw = torch.clamp(p2d_raw, -max_normalized, max_normalized)

    # Radial + tangential distortion
    k1 = dist[:, 0:1]; k2 = dist[:, 1:2]
    p1 = dist[:, 2:3]; p2 = dist[:, 3:4]
    k3 = dist[:, 4:5] if dist.shape[1] > 4 else torch.zeros_like(k1)

    x = p2d_raw[..., 0]; y = p2d_raw[..., 1]
    r2 = x*x + y*y;  r4 = r2*r2;  r6 = r4*r2
    kscale = 1 + k1*r2 + k2*r4 + k3*r6
    dx = 2*p1*x*y + p2*(r2 + 2*x*x)
    dy = p1*(r2 + 2*y*y) + 2*p2*x*y
    p2d_dist = kscale[..., None] * p2d_raw + torch.stack([dx, dy], dim=-1)

    # Apply intrinsics: p2d = p2d_dist @ K[:2,:2].T + K[:2,2] - offset
    p2d = esum(p2d_dist, mat[:, :2, :2], 'c m r, c i r -> c m i')
    p2d = p2d + rearrange(mat[:, :2, 2], 'c i -> c 1 i')
    p2d = p2d - rearrange(offset, 'c i -> c 1 i')

    out_shape = (ext.shape[0],) + orig_shape[:-1] + (2,)
    return p2d.reshape(out_shape)


def undistort_points_batched(mat, dist, offset, points):
    """Inverse of the distortion model via 5-step Newton-Raphson.

    Args:
        mat:    [cams, 3, 3]
        dist:   [cams, D]     (D >= 5)
        offset: [cams, 2]
        points: [cams, ..., 2]

    Returns:
        [cams, ..., 2]
    """
    cams = mat.shape[0]
    orig_shape = points.shape
    pts = rearrange(points, 'c ... r -> c (...) r')  # [cams, M, 2]

    fx = mat[:, 0, 0:1];  fy = mat[:, 1, 1:2]
    cx = mat[:, 0, 2:3];  cy = mat[:, 1, 2:3]

    x = (pts[:, :, 0] + offset[:, 0:1] - cx) / fx   # [cams, M]
    y = (pts[:, :, 1] + offset[:, 1:2] - cy) / fy
    x0, y0 = x.clone(), y.clone()

    k1 = dist[:, 0:1];  k2 = dist[:, 1:2]
    p1 = dist[:, 2:3];  p2 = dist[:, 3:4]
    k3 = dist[:, 4:5] if dist.shape[1] > 4 else torch.zeros_like(k1)

    for _ in range(5):
        r2 = x*x + y*y;  r4 = r2*r2;  r6 = r4*r2
        radial = 1 + k1*r2 + k2*r4 + k3*r6
        dx = 2*p1*x*y + p2*(r2 + 2*x*x)
        dy = p1*(r2 + 2*y*y) + 2*p2*x*y
        x = (x0 - dx) / radial
        y = (y0 - dy) / radial

    return torch.stack([x, y], dim=-1).reshape(orig_shape)


def is_point_visible_batched(size, ext, mat, dist, offset, p3d, margin=0):
    """Per-camera visibility as a boolean mask (no Python branching).

    Args:
        size:   [cams, 2]   (W, H)
        ext:    [cams, 4, 4]
        mat:    [cams, 3, 3]
        dist:   [cams, D]
        offset: [cams, 2]
        p3d:    [M, 3]

    Returns:
        visible: [cams, M] bool
    """
    p2d = project_cam_batched(ext, mat, dist, offset, p3d)   # [cams, M, 2]
    W = rearrange(size[:, 0].to(p2d.dtype), 'c -> c 1')
    H = rearrange(size[:, 1].to(p2d.dtype), 'c -> c 1')

    in_bounds = (
        (p2d[..., 0] >= margin) & (p2d[..., 0] < W - margin) &
        (p2d[..., 1] >= margin) & (p2d[..., 1] < H - margin)
    )   # [cams, M]

    p_cam = esum(to_homogeneous(p3d), ext, 'm r, c x r -> c m x')  # [cams, M, 4]
    in_front = p_cam[..., 2] > 0

    return in_bounds & in_front


def points_to_rays_batched(ext, mat, dist, offset, p2d, cube_scale):
    """Convert projected 2D points to ray-space 4×4 matrices (all cameras at once).

    Args:
        ext:        [cams, 4, 4]
        mat:        [cams, 3, 3]
        dist:       [cams, D]
        offset:     [cams, 2]
        p2d:        [cams, M, 2]
        cube_scale: scalar tensor

    Returns:
        ray_matrices: [cams, M, 4, 4]
    """
    cams, M = p2d.shape[:2]
    device, dtype = p2d.device, p2d.dtype

    p2d_und = undistort_points_batched(mat, dist, offset, p2d)   # [cams, M, 2]
    d_cam = to_homogeneous(p2d_und)                               # [cams, M, 3]
    d_cam = F.normalize(d_cam, dim=-1, eps=1e-8)

    R_c2w = rearrange(ext[:, :3, :3], 'c i j -> c j i')          # [cams, 3, 3]
    t_world = ext[:, :3, 3]
    t_c2w = -esum(R_c2w, t_world / cube_scale / 200.0, 'c i j, c j -> c i')  # [cams, 3]

    # Ray directions in world space
    d_world = esum(d_cam, R_c2w, 'c m j, c i j -> c m i')
    d_world = F.normalize(d_world, dim=-1, eps=1e-8)              # [cams, M, 3]

    cam_y  = rearrange(R_c2w[:, :, 1], 'c j -> c 1 j').expand(-1, M, -1)
    origin = rearrange(t_c2w, 'c j -> c 1 j').expand(-1, M, -1)

    z_ray = d_world
    x_ray = F.normalize(torch.cross(cam_y, z_ray, dim=-1), dim=-1, eps=1e-8)
    y_ray = F.normalize(torch.cross(z_ray, x_ray, dim=-1), dim=-1, eps=1e-8)

    R_w2r = torch.stack([x_ray, y_ray, z_ray], dim=-2)           # [cams, M, 3, 3]
    t_w2r = -esum(R_w2r, origin, 'c m i j, c m j -> c m i')      # [cams, M, 3]

    zeros = torch.zeros(cams, M, 1, 3, device=device, dtype=dtype)
    ones  = torch.ones(cams, M, 1, 1, device=device, dtype=dtype)
    top    = torch.cat([R_w2r, rearrange(t_w2r, 'c m i -> c m i 1')], dim=-1)
    bottom = torch.cat([zeros, ones], dim=-1)
    return torch.cat([top, bottom], dim=-2)                       # [cams, M, 4, 4]


def _masked_median_1d(values, mask):
    """Median of values[mask.bool()] via sort+gather. No .item(), no Python branches.

    Args:
        values: [N]  float
        mask:   [N]  float (0/1)

    Returns:
        scalar tensor  (1.0 fallback when mask is all-zero)
    """
    N = values.shape[0]
    v_sort = torch.sort(values * mask + (1.0 - mask) * 1e10).values  # [N] asc
    count = mask.sum().long()
    idx = ((count - 1) // 2).clamp(min=0, max=N - 1)
    median = torch.gather(v_sort.unsqueeze(0), 1,
                          idx.reshape(1, 1)).squeeze()  # scalar
    any_vis = (count > 0).float()
    return median * any_vis + (1.0 - any_vis)


def get_camera_scale_export(ext, mat, dist, offset, size, coords):
    """Export-safe replacement for get_camera_scale (matches median behaviour).

    Mirrors get_camera_scale exactly:
      • per-camera: median of singular-value sensitivities for visible points
      • cross-camera: median of per-camera medians
    No .item() call; returns a 0-d tensor.

    Args:
        ext:    [cams, 4, 4]
        mat:    [cams, 3, 3]
        dist:   [cams, D]
        offset: [cams, 2]
        size:   [cams, 2]   (W, H)
        coords: [M, 3]

    Returns:
        scale: scalar tensor
    """
    p = coords.float()
    cams = ext.shape[0]
    M    = p.shape[0]

    visible = is_point_visible_batched(size, ext, mat, dist, offset, p).float()  # [cams, M]

    fx = mat[:, 0, 0]; fy = mat[:, 1, 1]
    p_cam = esum(to_homogeneous(p), ext.float(), 'm r, c x r -> c m x')
    X = p_cam[..., 0]; Y = p_cam[..., 1]
    Z = p_cam[..., 2].clamp(min=1e-6)

    J00 = (rearrange(fx, 'c -> c 1') / Z).double()
    J02 = (-rearrange(fx, 'c -> c 1') * X / Z**2).double()
    J11 = (rearrange(fy, 'c -> c 1') / Z).double()
    J12 = (-rearrange(fy, 'c -> c 1') * Y / Z**2).double()
    zeros64 = torch.zeros_like(J00)

    J_proj = torch.stack([
        torch.stack([J00, zeros64, J02], dim=-1),
        torch.stack([zeros64, J11, J12], dim=-1),
    ], dim=-2)   # [cams, M, 2, 3]

    R = ext[:, :3, :3].double()
    J = esum(J_proj, R, 'c m i j, c j k -> c m i k')   # [cams, M, 2, 3]

    JJT = esum(J, J, 'c m i k, c m j k -> c m i j')
    a   = JJT[..., 0, 0]; b_v = JJT[..., 0, 1]; c_v = JJT[..., 1, 1]
    disc = ((a - c_v) * 0.5) ** 2 + b_v ** 2
    max_eig = (a + c_v) * 0.5 + torch.sqrt(disc.clamp(min=0.0))
    s = torch.sqrt(max_eig.clamp(min=0.0)).float()      # [cams, M]

    # Per-camera batched median — fully vectorized, no Python loop over cams.
    # 1e10 safely exceeds any realistic projection sensitivity (pixels/unit).
    s_for_sort = s * visible + (1.0 - visible) * 1e10   # push invisible → large
    s_sorted   = torch.sort(s_for_sort, dim=-1).values         # [cams, M] ascending
    count_vis  = visible.sum(dim=-1).long()                    # [cams]
    median_idx = ((count_vis - 1) // 2).clamp(min=0, max=M - 1)  # [cams]
    per_cam_raw = torch.gather(s_sorted, 1, median_idx.unsqueeze(1)).squeeze(1)  # [cams]
    any_vis_cam = (count_vis > 0).float()                      # [cams]
    per_cam = per_cam_raw * any_vis_cam + (1.0 - any_vis_cam)  # fallback 1.0

    # Cross-camera median over cams that have at least one visible point.
    any_vis = any_vis_cam                                      # [cams]
    sensitivity = _masked_median_1d(per_cam, any_vis)          # scalar

    return (1.0 / sensitivity.clamp(min=1e-8))
