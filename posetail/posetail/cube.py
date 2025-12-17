import itertools

import numpy as np

import torch
import torch.nn.functional as F

from einops import rearrange


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

def from_homogeneous(p, eps=1e-9):
    return p[..., :-1] / (p[..., -1, None] + eps) 

@torch.compile
def project_cam(cam, p3d_t):
    # p3d_t = torch.as_tensor(p3d)
    # ext_t = torch.as_tensor(cam.get_extrinsics_mat(), dtype=p3d_t.dtype, device=p3d_t.device)
    # mat_t = torch.as_tensor(cam.get_camera_matrix(), dtype=p3d_t.dtype, device=p3d_t.device)
    ext_t = cam['ext']
    mat_t = cam['mat']
    dist = cam['dist']

    p2d_proj_raw = torch.matmul(to_homogeneous(p3d_t), ext_t.T)
    p2d_proj_raw = from_homogeneous(p2d_proj_raw[..., :3])

    # k1, k2, p1, p2, k3 = dist
    # k4 = k5 = k6 = 0
    # r2 = torch.sum(torch.square(p2d_proj_raw), axis=1)
    # kscale = (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) / (1 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2)
    # #TODO: add p1, p2 effect
    # p2d_dist = kscale[:, None] * p2d_proj_raw

    p2d_dist = p2d_proj_raw
    
    p2d_raw = torch.matmul(to_homogeneous(p2d_dist), mat_t.T)
    p2d = from_homogeneous(p2d_raw)

    #TODO: handle offset
    
    return p2d

@torch.compile
def project_points_torch(camera_group, p3d):
    return torch.stack([project_cam(cam, p3d) 
                        for cam in camera_group])


class UnprojectViews:

    def __init__(self, 
                 camera_group, 
                 cube_center,
                 cube_extent = None,
                 cube_dim = 64, 
                 offset_dict = None, 
                 downsample_factor = 2, 
                 device = None):

        self.cgroup = camera_group
        self.cube_center = cube_center.cpu()
        self.cube_extent = cube_extent
        self.cube_dim = cube_dim 
        self.offset_dict = offset_dict
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


    # def get_cgroup_box(self):

    #     p3ds = []

    #     for a, b in itertools.combinations(range(len(self.cgroup.cameras)), 2):

    #         cgroup_sub = self.cgroup.subset_cameras([a, b])
    #         pts = []

    #         for cam in cgroup_sub.cameras:

    #             width, height = cam.get_size()
    #             pcam = np.array(
    #                 list(itertools.product(
    #                 np.linspace(0, width, num = 5),
    #                 np.linspace(0, height, num = 5)))
    #             )
    #             # mid = np.array(cam.get_size())/2
    #             pts.append(pcam)

    #         pts = np.array(pts)
    #         p3d = cgroup_sub.triangulate(pts)
    #         p3ds.append(p3d)

    #     p3ds = np.vstack(p3ds)
    #     crange = np.diff(np.percentile(p3ds, [5, 95], axis = 0), axis = 0)

    #     center = np.median(p3ds, axis = 0)
    #     cube_extent = np.max(crange) / 2

    #     return center, cube_extent


    def create_mesh_3d(self):

        xs = self.init_coords(dim = 0)
        ys = self.init_coords(dim = 1)
        zs = self.init_coords(dim = 2)

        # create a mesh of all coords in the volume
        coords = np.array(np.meshgrid(zs, xs, ys))
        coords_flat = rearrange(coords, 'r cd ch cw -> (cd ch cw) r')
        coords_flat = torch.from_numpy(coords_flat).to(self.device).float()

        # project coordinates for each camera 
        coords_proj = project_points_torch(self.cgroup, coords_flat) / self.downsample_factor
        camera_names = [cam['name'] for cam in self.cgroup]
                
        # account for camera cropping # TODO: test this once we have a dataset to do so
        if self.offset_dict: 
            offsets = torch.tensor([self.offset_dict[name]] for name in camera_names).float()
            coords_proj = coords - offsets[:, None, :] / self.downsample_factor

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
