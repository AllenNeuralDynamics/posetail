import itertools

import numpy as np

import torch
import torch.nn.functional as F

from einops import rearrange, einsum, repeat


def project_volumes(volumes): 
    ''' 
    project volume to get the xy, xz, and yz planes 
    ''' 
    xy_planes = torch.sum(volumes, dim = -1)
    xz_planes = torch.sum(volumes, dim = -2)
    yz_planes = torch.sum(volumes, dim = -3)

    return xy_planes, xz_planes, yz_planes


def to_homogeneous(p):
    one_size = p.shape[:-1] + (1,)
    ones = torch.ones(size=one_size, dtype=p.dtype, device=p.device)
    return torch.cat([p, ones], dim=-1)


def from_homogeneous(p, eps=1e-10):
    denom = p[..., -1, None]
    denom = torch.where(denom >= 0, 
                        torch.clamp(denom, min=eps), 
                        torch.clamp(denom, max=-eps))
    return p[..., :-1] / denom    
    # return p[..., :-1] / (p[..., -1, None] + eps) 


# @torch.compile
def project_cam(cam, p3d_t, downsample_factor = 1, max_normalized = 3.0):
    # p3d_t = torch.as_tensor(p3d)
    # ext_t = torch.as_tensor(cam.get_extrinsics_mat(), dtype=p3d_t.dtype, device=p3d_t.device)
    # mat_t = torch.as_tensor(cam.get_camera_matrix(), dtype=p3d_t.dtype, device=p3d_t.device)
    ext_t = cam['ext']
    mat_t = cam['mat']
    dist = cam['dist']
    cam_type = cam['type'] # pinhole, fisheye # TODO: implement functionality for different camera types

    p2d_proj_raw = torch.matmul(to_homogeneous(p3d_t), ext_t.T)
    p2d_proj_raw = from_homogeneous(p2d_proj_raw[..., :3])

    # handle points way outside of the frame
    p2d_proj_raw = torch.clamp(p2d_proj_raw, -max_normalized, max_normalized)
    
    k1, k2, p1, p2, k3 = dist[:5]
    k4 = k5 = k6 = 0
    r2 = torch.sum(torch.square(p2d_proj_raw), dim=-1)
    r4 = r2 * r2
    r6 = r4 * r2
    kscale = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)

    x = p2d_proj_raw[..., 0]
    y = p2d_proj_raw[..., 1]
    dx = 2*p1*x*y + p2 * (r2 + 2*x*x)
    dy = p1*(r2 + 2*y*y) + 2*p2*x*y
    p1_p2_add = torch.stack([dx, dy], dim=-1)
    
    p2d_dist = kscale[..., None] * p2d_proj_raw + p1_p2_add

    # p2d_dist = p2d_proj_raw

    p2d = torch.matmul(p2d_dist, mat_t[:2,:2].T) + mat_t[:2,2]
    
    # p2d_raw = torch.matmul(to_homogeneous(p2d_dist), mat_t.T)
    # p2d = from_homogeneous(p2d_raw)

    # handle camera offset
    if 'offset' in cam: 
        offset = cam['offset']
        p2d = p2d - offset[None, :]

    # account for downsampling
    p2d = p2d / downsample_factor
    
    return p2d

# @torch.compile
def project_points_torch(camera_group, coords_3d, downsample_factor = 1):

    coords_proj = torch.stack([project_cam(cam, coords_3d, downsample_factor)
                               for cam in camera_group])

    return coords_proj


def triangulate_simple(points, camera_mats, weights):
    '''
    Inputs:
        points: [C, 2] 2d points to triangulate
        camera_mats: [C, 4, 4] camera extrinsics
        weights: [C] weight for each camera
    Outputs:
        p3d: [3] triangulated 3d point
    '''
    num_cams = len(camera_mats)
    A = torch.zeros((num_cams * 2, 4), dtype=points.dtype, device=points.device)
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        w = weights[i]
        A[(i * 2):(i * 2 + 1)] = w * (x * mat[2] - mat[0])
        A[(i * 2 + 1):(i * 2 + 2)] = w * (y * mat[2] - mat[1])
    u, s, vh = torch.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d

def triangulate_simple_batch(points, camera_mats, weights):
    '''
    Inputs:
        points: [C, N, 2] 2d points to triangulate
        camera_mats: [C, 4, 4] camera extrinsics
        weights: [C, N] weight for each camera
    Outputs:
        p3d: [N, 3] triangulated 3d point
    '''
    C, N, _ = points.shape
    points = rearrange(points, 'c n r -> n c r')
    cam_mats = repeat(camera_mats, 'c i j -> n c i j', n=N)
    
    x = points[:, :, 0:1, None]
    y = points[:, :, 1:2, None]
    w = rearrange(weights, 'c n -> n c 1 1')
    
    eq_x = w * (x * cam_mats[:, :, 2:3, :] - cam_mats[:, :, 0:1, :])
    eq_y = w * (y * cam_mats[:, :, 2:3, :] - cam_mats[:, :, 1:2, :])
    
    A = rearrange([eq_x, eq_y], 'two n c 1 j -> n (c two) j')
    
    # Use eigendecomposition of A^T A instead of SVD
    # This is more numerically stable for gradients
    ATA = torch.bmm(A.transpose(-2, -1), A)
    
    # Add small regularization
    eps = 1e-6
    ATA = ATA + eps * torch.eye(4, device=A.device, dtype=A.dtype)
    
    # Smallest eigenvalue's eigenvector
    eigenvalues, eigenvectors = torch.linalg.eigh(ATA)
    p3d_homogeneous = eigenvectors[:, :, 0]  # eigenvector for smallest eigenvalue
    
    p3d = from_homogeneous(p3d_homogeneous)
    
    return p3d



def undistort_points(cam, points):
    matrix = cam['mat']
    dist = cam['dist']
    offset = cam['offset']

    shape = points.shape
    points = points.reshape(-1, 2)
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    x = (points[:, 0] + offset[0] - cx) / fx
    y = (points[:, 1] + offset[1] - cy) / fy
    x0, y0 = x.clone(), y.clone()
    for _ in range(5):
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2
        k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]
        if dist.shape[0] > 4:
            k3 = dist[4]
        else:
            k3 = torch.tensor(0.0, device=dist.device, dtype=dist.dtype)
        radial = 1 + k1*r2 + k2*r4 + k3*r6
        dx = 2*p1*x*y + p2*(r2 + 2*x*x)
        dy = p1*(r2 + 2*y*y) + 2*p2*x*y
        x = (x0 - dx) / radial
        y = (y0 - dy) / radial
        
    return torch.stack([x, y], dim=1).reshape(shape)


def projection_sensitivity(cam, p):
    p = p.float()
    n_points = p.shape[0]
    fx = cam['mat'][0,0]
    fy = cam['mat'][1,1]
    ext_t = cam['ext'].float()
    
    p_cam = torch.matmul(to_homogeneous(p), ext_t.T)[:, :3]
    depth = p_cam[:,2]
    # print("depth min:", depth.min())
    # print("depth max:", depth.max())
    # print("near zero:", (depth.abs() < 1e-6).sum())
    # print("negative:", (depth < 0).sum())
    X = p_cam[:, 0]
    Y = p_cam[:, 1]
    Z = p_cam[:, 2]
    
    J_proj = torch.zeros((n_points, 2, 3), dtype=torch.float64, device=p_cam.device)
    J_proj[:, 0, 0] = fx / Z
    J_proj[:, 0, 2] = -fx * X / (Z**2)
    J_proj[:, 1, 1] = fy / Z
    J_proj[:, 1, 2] = -fy * Y / (Z**2)
    
    R = ext_t[:3,:3].to(torch.float64)
    J = einsum(J_proj, R, 'n i j, j k -> n i k')
    return J

def is_point_visible(cam, p3d, margin=0):
    """
    Check if 3D points project into camera view.
    margin: pixels from border (e.g., 10 to avoid edge effects)
    """
    p2d = project_cam(cam, p3d)
    w, h = cam['size']

    # check if in bounds
    in_bounds = (
        (p2d[:, 0] >= margin) & 
        (p2d[:, 0] < w - margin) &
        (p2d[:, 1] >= margin) & 
        (p2d[:, 1] < h - margin)
    )

    # check if point is in front of camera
    p_cam = torch.matmul(to_homogeneous(p3d), cam['ext'].T)[:, :3]
    in_front = p_cam[:, 2] > 0

    return in_bounds & in_front

def get_camera_scale(camera_group, p):

    ps = []

    for cam in camera_group:
        visible = is_point_visible(cam, p)

        if torch.sum(visible) > 0:

            with torch.autocast(device_type = p.device.type, enabled = False):
                J = projection_sensitivity(cam, p[visible])
                s = torch.linalg.svdvals(J.float())[:, 0]
                
            ps.append(torch.median(s))

    if len(ps) == 0:
        return torch.nan
    
    sensitivity = torch.median(torch.as_tensor(ps))
    scale = 1 / sensitivity

    return scale.item()

class UnprojectViews:

    def __init__(self, 
                 camera_group, 
                 cube_center,
                 cube_extent = None,
                 cube_dim = 64, 
                 downsample_factor = 2, 
                 device = None):

        self.cgroup = camera_group
        self.cube_center = cube_center.cpu()
        self.cube_extent = cube_extent
        self.cube_dim = cube_dim 
        self.downsample_factor = downsample_factor
        self.device = device

        if cube_extent is not None: 
            self.cube_extent = cube_extent

        self.coords_proj = self.create_mesh_3d()


    def init_coords(self, dim = 0): 

        coords = np.linspace(
            self.cube_center[dim] - self.cube_extent, 
            self.cube_center[dim] + self.cube_extent, 
            num = self.cube_dim
        )

        return coords
    

    def create_mesh_3d(self):

        xs = self.init_coords(dim = 0)
        ys = self.init_coords(dim = 1)
        zs = self.init_coords(dim = 2)

        # create a mesh of all coords in the volume
        coords = np.array(np.meshgrid(zs, xs, ys))
        coords_flat = rearrange(coords, 'r cd ch cw -> (cd ch cw) r')
        coords_flat = torch.from_numpy(coords_flat).to(self.device).float()

        # project coordinates for each camera 
        coords_proj = project_points_torch(
            camera_group = self.cgroup, 
            coords_3d = coords_flat,
            downsample_factor = self.downsample_factor, 
        )

        return coords_proj


    def unproject_to_volume(self, feature_maps): 

        sampled_points = []

        for i, (features, coord_grid) in enumerate(zip(feature_maps, self.coords_proj)):

            B, S, D, H, W = features.shape
            features = rearrange(features, 'b s d h w -> (b s) d h w')

            coord_grid = rearrange(coord_grid, "dhw r -> 1 1 dhw r") 
            coord_grid = (torch.as_tensor(coord_grid)
                               .repeat((B * S, 1, 1, 1))
                               .to(features.device))

            scale = torch.tensor([H, W], device = features.device) 
            coord_grid = 2 * coord_grid / scale - 1

            # sample projected volumetric coordinates from the feature maps 
            sampled = F.grid_sample(
                features.float(), 
                coord_grid.float(),
                mode = 'bilinear', 
                padding_mode = 'zeros',
                align_corners = True
            )

            # switch from d h w -> h w d
            sampled = rearrange(
                sampled, 
                '(b s) d 1 (cd ch cw) -> 1 b s d ch cw cd', 
                b = B, s = S,
                ch = self.cube_dim, 
                cw = self.cube_dim, 
                cd = self.cube_dim
            )

            sampled_points.append(sampled)

        # (n_cams, B, S, D, cube_dim, cube_dim, cube_dim)
        volumes = torch.vstack(sampled_points) 

        return volumes
