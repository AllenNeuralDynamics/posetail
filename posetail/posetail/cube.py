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


class UnprojectViews:

    def __init__(self, 
                 camera_group, 
                 cube_dim = 64, 
                 cube_extent = None,
                 offset_dict = None, 
                 downsample_factor = 2, 
                 device = None):

        self.cgroup = camera_group
        self.cube_dim = cube_dim 
        self.offset_dict = offset_dict
        self.downsample_factor = downsample_factor
        self.device = device

        self.cube_center, self.cube_extent = self.get_cgroup_box()

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


    def get_cgroup_box(self):

        p3ds = []

        for a, b in itertools.combinations(range(len(self.cgroup.cameras)), 2):

            cgroup_sub = self.cgroup.subset_cameras([a, b])
            pts = []

            for cam in cgroup_sub.cameras:

                width, height = cam.get_size()
                pcam = np.array(
                    list(itertools.product(
                    np.linspace(0, width, num = 5),
                    np.linspace(0, height, num = 5)))
                )
                # mid = np.array(cam.get_size())/2
                pts.append(pcam)

            pts = np.array(pts)
            p3d = cgroup_sub.triangulate(pts)
            p3ds.append(p3d)

        p3ds = np.vstack(p3ds)
        crange = np.diff(np.percentile(p3ds, [5, 95], axis = 0), axis = 0)

        center = np.median(p3ds, axis = 0)
        cube_extent = np.max(crange) / 2

        return center, cube_extent


    def create_mesh_3d(self):

        xs = self.init_coords(dim = 0)
        ys = self.init_coords(dim = 1)
        zs = self.init_coords(dim = 2)

        # create a mesh of all coords in the volume
        coords = np.array(np.meshgrid(zs, xs, ys))
        coords_flat = rearrange(coords, 'r cd ch cw -> (cd ch cw) r')

        # project coordinates for each camera 
        coords_proj = list(self.cgroup.project(coords_flat))
        camera_names = self.cgroup.get_names()

        # account for camera cropping
        if self.offset_dict: 

            for i, (c_name, coords) in enumerate(zip(camera_names, coords_proj)): 
                xy = np.array(self.offset_dict[c_name])
                coords_proj[i] = (coords - xy) / self.downsample_factor

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