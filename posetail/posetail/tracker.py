import itertools
import timm

import torch
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange
from frozendict import frozendict, deepfreeze

from posetail.posetail.cube import UnprojectViews, project_volumes
from posetail.posetail.transformer import TimeSpaceTransformer, MLP
from posetail.posetail.networks import FeatureExtractor, ResidualFeatureExtractor, TriplaneFeatureExtractor
from posetail.posetail.utils import get_pos_encoding, get_fourier_encoding


def replace_norm(model):
    
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            instance_norm = nn.InstanceNorm2d(child.num_features)
            setattr(model, name, instance_norm)
        else:
            replace_norm(child)


class Tracker(nn.Module): 

    def __init__(self, track_3d = True, stride_length = 8, 
                 stride_overlap = None, downsample_factor = 4, 
                 cube_dim = 20, cube_extent = 200, 
                 upsample_factor = 1, corr_levels = 4, 
                 corr_radius = 3, corr_hidden_dim = 384,
                 corr_output_dim = 256, max_freq = 10, 
                 n_iters = 4, embedding_dim = 256, 
                 latent_dim = 128, encoding_dim = 64,
                 n_virtual = 64, n_heads = 8, 
                 n_time_space_blocks = 6, embedding_factor = 4,
                 device = None): 
        super().__init__()

        self.device = device

        if track_3d: 
            self.R = 3 
            self.unproject_groups = {}
        else: 
            self.R = 2

        # video processing
        self.S = stride_length
        
        if stride_overlap is None: 
            self.stride_overlap = self.S // 2
        else:
            self.stride_overlap = stride_overlap 
        
        # cnn params
        self.downsample_factor = downsample_factor
        self.latent_dim = latent_dim 

        # cube params
        self.cube_dim = cube_dim 
        self.cube_extent = cube_extent

        # this gives us the indices for xy, xz, and yz planes for 3d 
        # tracking, or xy planes for 2d tracking
        self.plane_ixs = list(itertools.combinations(range(self.R), 2))
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

        # add up the dimensions for transformer input
        self.input_dim = (2 + 2 * self.R + 4 * self.R * self.max_freq +
                          self.corr_levels * self.corr_output_dim)
        # print(f'transformer input dimension: {self.input_dim}') 

        # networks
        self.cnn = ResidualFeatureExtractor(
            input_dim = 3, # RGB 
            output_dim = self.latent_dim,
            n_blocks = 4,
            kernel_size = 3,
            downsample_factor = self.downsample_factor,
            spatial_res_factor = 2 
        )

        if self.R == 3: 

            # triplane features
            self.triplane_cnn = TriplaneFeatureExtractor(
                input_dim = self.latent_dim * len(self.plane_ixs), 
                n_hidden_layers = 2, 
                kernel_size = 3, 
                padding = 1, 
                upsample_factor = self.upsample_factor
            )

        # correlation features
        self.corr_mlp = MLP(
            input_dim = (2 * self.corr_radius + 1) ** 4, 
            embedding_dim = self.corr_hidden_dim, 
            output_dim = self.corr_output_dim
        )

        # time embeddings
        t = torch.arange(self.S, device = self.device)

        self.register_buffer(
            'time_encoding', 
            get_pos_encoding(t, self.input_dim)
        )

        # transformer
        self.tsformer = TimeSpaceTransformer(
            input_dim = self.input_dim, 
            embedding_dim = self.embedding_dim, 
            output_dim = self.R + 2, 
            n_time_space_blocks = self.n_time_space_blocks, 
            n_heads = self.n_heads,
            n_virtual = self.n_virtual, 
            embedding_factor = self.embedding_factor, 
            vc_head = True, 
            device = self.device,
            **self.activation_kwargs
        )


    def init_stride(self, pred, prev_ix):

        stride_remainder = self.S - self.stride_overlap
        expansion = [1 for x in pred.shape]
        expansion[1] = stride_remainder

        first = pred[:, prev_ix:prev_ix + self.stride_overlap, ...]
        last = first[:, -1:, ...].repeat(expansion)
        init = torch.cat((first, last), dim = 1)

        return init

    def init_levels(self, coords, feature_planes, dims):

        feature_planes_levels = []
        track_features_levels = []

        B, T = dims

        for i, corr_level in enumerate(range(1, self.corr_levels + 1)): 

            # downsize image features according to corr level
            if corr_level > 1: 

                feature_planes_scaled = rearrange(
                    feature_planes_scaled, 
                    'b t d v1 v2 r -> (b t) (d r) v1 v2' 
                )

                feature_planes_scaled = F.avg_pool2d(
                    input = feature_planes_scaled, 
                    kernel_size = 2, 
                    stride = 2
                ) 
            else: 
                feature_planes_scaled = feature_planes

            feature_planes_scaled = rearrange(
                feature_planes_scaled, 
                '(b t) (d r) v1 v2 -> b t d v1 v2 r', 
                b = B, t = T, r = len(self.plane_ixs)
            )

            track_features = []

            for k, ixs in enumerate(self.plane_ixs):

                tf = self.get_track_features(
                    coords = coords[..., ixs] / 2**i, 
                    feature_planes = feature_planes_scaled[..., k]
                )
                track_features.append(tf)

            feature_planes_levels.append(feature_planes_scaled)
            track_features = torch.stack(track_features, axis = -1)
            track_features_levels.append(track_features)

        return feature_planes_levels, track_features_levels

    
    def get_coord_regions(self, coords):

        B, _, N, R = coords.shape
        coords = rearrange(coords, 'b 1 n r -> b n 1 1 r')

        offsets = torch.arange(self.corr_dim, device = self.device) - self.corr_radius

        offset_coords = torch.tensor(list(itertools.product(offsets, offsets)), device = self.device)
        offset_coords = offset_coords.T.reshape((R - 1, self.corr_dim, self.corr_dim))

        time_offsets = torch.zeros((1, self.corr_dim, self.corr_dim), device = self.device)
        offset_coords = torch.cat((offset_coords, time_offsets), dim = 0)
        offset_coords = rearrange(offset_coords, 'r ch cw -> ch cw r')

        coord_regions = coords + offset_coords
        coord_regions = rearrange(coord_regions, 'b n ch cw r -> b (ch cw) n r')

        return coord_regions


    def get_track_features(self, coords, feature_planes, 
                           padding_mode = 'border'): 

        _, N, R = coords.shape
        B, S, D, V1, V2 = feature_planes.shape

        feature_planes = rearrange(feature_planes, 'b s d vh vw -> b d s vh vw')

        t = torch.arange(B, device = self.device) * torch.ones((N, B), device = self.device) 
        t = rearrange(t, 'n b -> b 1 n 1')

        coords = rearrange(coords, 'b n r -> b 1 n r')
        coords = torch.cat((coords, t), dim = -1)
        coord_regions = self.get_coord_regions(coords)
    
        scale, _ = (torch.max(torch.tensor([V2, V1, S], device = self.device)
                          .unsqueeze(dim = -1) - 1,
                          dim = 1))

        coord_regions = rearrange(coord_regions, 'b corr n r -> b corr n 1 r')
        coord_regions = coord_regions * (2 / scale) - 1

        track_features = F.grid_sample(
            input = feature_planes.float(),
            grid = coord_regions.float(), 
            align_corners = True,
            padding_mode = padding_mode
        )

        track_features = rearrange(track_features, 'b d corr n 1 -> b corr n d')

        return track_features


    def get_corr_features_2d(self, coords, feature_planes, 
                             padding_mode = 'border'): 

        _, N, R = coords.shape # B * S, N, R
        B, S, D, V1, V2 = feature_planes.shape

        times = torch.zeros((B * S, N, 1), device = self.device)
        coords = torch.cat((coords, times), dim = -1)

        coord_regions = self.get_coord_regions(coords.unsqueeze(1))

        coord_regions = rearrange(
            coord_regions, 
            'bs (corr1 corr2) n r -> bs n corr1 corr2 r', 
            corr1 = self.corr_dim, corr2 = self.corr_dim
        )

        scale, _ = (torch.max(torch.tensor([V2, V1, S], device = self.device)
                          .unsqueeze(dim = -1) - 1,
                          dim = 1))
        coord_regions = coord_regions * (2 / scale) - 1

        feature_planes = rearrange(
            feature_planes, 
            'b s d vh vw -> (b s) d 1 vh vw'
        )

        corr_features = F.grid_sample(
            input = feature_planes.float(),
            grid = coord_regions.float(), 
            align_corners = True,
            padding_mode = padding_mode
        )

        corr_features = rearrange(
            corr_features, 
            '(b s) d n corr1 corr2 -> b s n corr1 corr2 d', 
            b = B, s = S, corr1 = self.corr_dim, corr2 = self.corr_dim
        )

        return corr_features

    
    def get_corr_features_4d(self, coords, feature_planes, track_features): 

        corr_features_2d = self.get_corr_features_2d(
            coords = coords, 
            feature_planes = feature_planes
        ) 
        
        track_features = rearrange(
            track_features, 
            'b (ch cw) n d -> b n d ch cw', 
            ch = self.corr_dim, 
            cw = self.corr_dim
        )

        corr_features_4d = torch.einsum(
            'bsnhwd,bndxy->bsnhwxy', 
            corr_features_2d, 
            track_features
        )

        corr_features_4d = rearrange(
            corr_features_4d, 
            'b s n ch cw cx cy -> (b s n) (ch cw cx cy)'
        )
                                    
        return corr_features_4d


    def get_corr_features(self, coords, feature_planes_levels, track_features_levels): 

        B, S, N, R = coords.shape
        corr_features_levels = []

        for i in range(self.corr_levels):

            corr_features = []

            for j, ixs in enumerate(self.plane_ixs):
                
                coords_scaled = rearrange(
                        coords[..., ixs] / 2**i, 
                        'b s n r -> (b s) n r'
                )

                corr_features_4d = self.get_corr_features_4d(
                    coords = coords_scaled, 
                    feature_planes = feature_planes_levels[i][..., j], 
                    track_features = track_features_levels[i][..., j]
                )

                corr_features.append(self.corr_mlp(corr_features_4d))

            # sum xy, xz, and yz correlation features
            cf = torch.sum(torch.stack(corr_features, dim = -1), dim = -1)
            corr_features_levels.append(cf)

        corr_features = rearrange(
            torch.cat(corr_features_levels,  dim = -1), 
            '(b s n) d -> b s n d', 
            b = B, s = S, n = N
        )

        return corr_features


    def forward_iteration(self, coords, vis, conf, 
                          feature_planes_levels, 
                          track_features_levels):

        B, S, D, V1, V2, R = feature_planes_levels[0].shape
        B, S, N, R = coords.shape

        # iterative updates
        coords_pred = []
        vis_pred = []
        conf_pred = []

        for i in range(self.n_iters):

            # correlation features for transformer input
            corr_features = self.get_corr_features(
                coords = coords, 
                feature_planes_levels = feature_planes_levels, 
                track_features_levels = track_features_levels
            )

            # encodings for transformer input
            forward_flow = torch.diff(coords, dim = 1)
            backward_flow = (torch.diff(coords.flip(dims = [1]), dim = 1)
                                  .flip(dims = [1]))

            forward_flow = F.pad(
                forward_flow, 
                pad = (0, 0, 0, 0, 1, 0), 
                mode = 'constant', 
                value = 0
            ) 

            backward_flow = F.pad(
                backward_flow, 
                pad = (0, 0, 0, 0, 0, 1), 
                mode = 'constant', 
                value = 0
            ) 

            scale = V1 
            if self.R == 2: 
                scale = (torch.tensor([V1, V2, V1, V2])
                              .view(1, 1, 1, -1)
                              .to(self.device))

            coord_flow = torch.cat([forward_flow, backward_flow], dim = -1) / scale
            flow_encoding = get_fourier_encoding(coord_flow, min_freq = 0, max_freq = self.max_freq)

            # transformer time
            x = (torch.cat([vis, conf, corr_features, coord_flow, flow_encoding], dim = -1) 
                           + self.time_encoding)  

            updates = self.tsformer(x) # b s n d

            # update coords, vis, and conf
            delta_coords, delta_vis, delta_conf = torch.split(updates, [self.R, 1, 1], dim = -1)
            coords = coords + delta_coords # b s n 3
            vis = torch.sigmoid(vis + delta_vis) # b s n 1 
            conf = torch.sigmoid(conf + delta_conf) # b s n 1 
 
            coords_pred.append(coords)
            vis_pred.append(vis) 
            conf_pred.append(conf)
        
        return coords_pred, vis_pred, conf_pred

    def forward(self, views, coords, camera_group = None, 
                offset_dict = None):
        '''
        B: batch size
        T: number of frames in video
        C: number of channels 
        H: height of image
        W: width of image
        D: latent dimension
        '''
        coords.to(self.device)

        B, N, R = coords.shape
        B, T, H, W, C = views[0].shape

        assert R == self.R

        if self.R == 3:
            # create a hash of frozen camera group dictionary
            cgroup_hash = hash(deepfreeze(camera_group.get_dicts()))

            if cgroup_hash in self.unproject_groups:
                # access stored camera group
                cgroup = self.unproject_groups[cgroup_hash]

            else:
                # TODO: check whether this can be created with 
                # torch.no_grad()
                # store unproject views for later
                with torch.no_grad():

                    cgroup = UnprojectViews(
                        camera_group = camera_group,
                        offset_dict = offset_dict,
                        downsample_factor = self.downsample_factor,
                        cube_dim = self.cube_dim, 
                        cube_extent = self.cube_extent, 
                        device = self.device) 

                    self.unproject_groups[cgroup_hash] = cgroup

        # normalize frames
        for i, frames in enumerate(views): 
            frames = 2 * (frames / 255.0) - 1
            views[i] = rearrange(frames, 'b t h w c -> b t c h w').to(self.device)

        # determine number of strides
        stride_remainder = self.S - self.stride_overlap
        n_windows = ((T - stride_remainder - 1) // stride_remainder) + 1
        n_pad = (self.S - T % self.S) % self.S

        # frames for each camera view
        if n_pad > 0: 
            for i, frames in enumerate(views):
                last_frame = frames[:, -1:, ...].repeat((1, n_pad, 1, 1, 1))
                frames = torch.cat((frames, last_frame), dim = 1)
                views[i] = frames

        # extract image features and downsize
        feature_maps = []

        for i, frames in enumerate(views):

            frames = rearrange(frames, 'b t c h w -> (b t) c h w')
            feature_map = self.cnn(frames)
            _, D, H2, W2 = feature_map.shape

            feature_maps.append(
                rearrange(feature_map, 
                '(b t) d h2 w2 -> b t d h2 w2',
                b = B, t = T + n_pad)
            )

        coords = coords / self.downsample_factor


        # NOTE: this can be put in the forward loop if its too memory 
        # intensive up front 
        if self.R == 3: 

            # unproject views into volumes, get average volume, then project
            #with torch.no_grad():
            volumes = cgroup.unproject_to_volume(feature_maps) # (G, B, S, D, V1, V2, V3)
            # torch.save(feature_maps, f'/home/katie.rupp/posetail/volumes/feature_maps.pt')
            # torch.save(volumes, f'/home/katie.rupp/posetail/volumes/volumes_iter.pt')
            volumes_avg = torch.mean(volumes, dim = 0) 

            xy_planes, xz_planes, yz_planes = project_volumes(volumes_avg)
            planes = torch.cat((xy_planes, xz_planes, yz_planes), dim = -3) 

            # extract features from planes 
            feature_planes = self.triplane_cnn(
                rearrange(planes, 'b t d3 v1 v2 -> (b t) d3 v1 v2')
            )

        else: # 2d case, no triplane
            feature_planes = rearrange(feature_maps, '1 b t d h w -> (b t) d h w')

        # initialize feature planes and track features for each correlation level
        (feature_planes_levels, 
         track_features_levels) = self.init_levels(coords = coords, 
                                                   feature_planes = feature_planes, 
                                                   dims = (B, T + n_pad))

        # track final predictions
        coords_pred = torch.zeros((B, T, N, R), device = self.device)
        vis_pred = torch.zeros((B, T, N, 1), device = self.device)
        conf_pred = torch.zeros((B, T, N, 1), device = self.device)

        # track iterative predictions
        coords_pred_iters = []
        vis_pred_iters = []
        conf_pred_iters = []

        for i in range(n_windows): 

            ix = stride_remainder * i
            prev_ix = ix - self.stride_overlap

            # get the frames for the current stride
            views_subset = [frames[:, ix:ix + self.S, ...] for frames in views]

            if i == 0:
                # initialize coords with the first frame in the window 
                # and the visibility of all tracks to 1 
                coords_init = coords.unsqueeze(1).repeat((1, self.S, 1, 1))
                vis_init = torch.zeros((B, self.S, N, 1), device = self.device)
                conf_init = torch.zeros((B, self.S, N, 1), device = self.device)

            else: 
                # copy overlapping coords, vis, and conf predictions from the
                # previous window to the the first frames in the current window 
                coords_init = self.init_stride(coords_pred, prev_ix)
                vis_init = self.init_stride(vis_pred, prev_ix)
                conf_init = self.init_stride(conf_pred, prev_ix)

            # get the feature planes and track features for the current stride
            feature_plane_levels_subset = [fp[:, ix:ix + self.S, ...] 
                                           for fp in feature_planes_levels]

            # iterative updates with transformer
            (coords_pred_updates, 
            vis_pred_updates, 
            conf_pred_updates) = self.forward_iteration(
                coords = coords_init, 
                vis = vis_init,
                conf = conf_init,
                feature_planes_levels = feature_plane_levels_subset, 
                track_features_levels = track_features_levels
            )

            # remove excess padding and store final iteration
            if i == n_windows - 1 and n_pad > 0:
                coords_pred_updates = [update[:, :-n_pad, ...] for update in coords_pred_updates]
                vis_pred_updates = [update[:, :-n_pad, ...] for update in vis_pred_updates]
                conf_pred_updates = [update[:, :-n_pad, ...] for update in conf_pred_updates]

            # store all iterative updates
            coords_pred[:, ix:ix + self.S, ...] = coords_pred_updates[-1] 
            vis_pred[:, ix:ix + self.S, ...] = vis_pred_updates[-1]
            conf_pred[:, ix:ix + self.S, ...] = conf_pred_updates[-1]

            coords_pred_iters.append([coords * self.downsample_factor for coords in coords_pred_updates])
            vis_pred_iters.append(vis_pred_updates)
            conf_pred_iters.append(conf_pred_updates)

        # adjust coordinates to match original scale
        coords_pred = coords_pred * self.downsample_factor 

        return coords_pred, vis_pred, conf_pred, coords_pred_iters, vis_pred_iters, conf_pred_iters