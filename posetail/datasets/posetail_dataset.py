import os 
import cv2
import json

import torch 
from torch.utils.data import Dataset

import numpy as np
import pandas as pd 

from aniposelib.cameras import CameraGroup, Camera
from easydict import EasyDict as edict
from einops import rearrange

from posetail.datasets.utils import get_dirs, load_yaml, disassemble_extrinsics, format_sample_input
from posetail.posetail.cube import project_points_torch, is_point_visible
from train_utils import format_camera_group, dict_to_device

from pprint import pprint

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='imagecorruptions')

import imgaug.augmenters as iaa

def custom_collate(batch):
    ''' 
    custom collate functon to enable returning 
    non-tensor, non-list, etc type objects from 
    the default collate function
    '''
    batch = list(zip(*batch))

    views = [torch.stack(v, dim = 0) for v in zip(*list(batch[0]))]
    fnums = torch.stack(batch[3], axis = 0)
    cgroup = batch[4][0]

    # mask nan coordinates in the first frame
    coords = torch.stack(batch[1], axis = 0)
    mask = torch.isfinite(coords).all(dim = -1).all(dim = 1).all(dim = 0)
    coords_masked = coords[:, :, mask, :]

    # get corresponding visibilities if present
    vis_masked = None
    if batch[2][0] is not None: 
        vis = torch.stack(batch[2], axis = 0)
        vis_masked = vis[:, :, mask].unsqueeze(-1)

    rows = batch[5][0]
        
    batch = edict({'views': views, 
                   'coords': coords_masked,
                   'vis': vis_masked,
                   'fnums': fnums,
                   'cgroup': cgroup, 
                   'sample_info': rows})

    return batch


class PosetailDataset(Dataset): 

    def __init__(self, config, split): 

        self.split = split
        assert split in {'train', 'val', 'test'}
        self.split_dir = config.dataset[split].get('split_dir')

        self.data_path = config.dataset.prefix
        self.n_frames = config.dataset[split].get('n_frames', 16)
        self.max_res = config.dataset[split].get('max_res', -1) # -1 means no resizing
        self.min_res = config.dataset[split].get('min_res', self.max_res) # only used when max_res != -1
        self.aug_prob = config.dataset[split].get('aug_prob', 0.25)

        self.crop_to_points = config.dataset[split].get('crop_to_points', True)
        self.min_crop_dim = config.dataset[split].get('min_crop_dim', 64)
        
        # for sampling cameras, keypoints
        self.cams_to_sample = format_sample_input(config.dataset[split].get('cams_to_sample', None))
        self.kpts_to_sample = format_sample_input(config.dataset[split].get('kpts_to_sample', None))
        self.cam_thresh_for_vis = config.dataset[split].get('cam_thresh_for_vis', 1) 
        self.enable_kpt_filtering = config.dataset[split].get('enable_kpt_filtering', False)
        
        # for balancing datasets
        self.balance_datasets = config.dataset[split].get('balance_datasets', True)
        self.n_samples_per_dataset = config.dataset[split].get('n_samples_per_dataset', -1) # default balances based on dataset with the most samples

        # augmentation
        self.aug = iaa.Sequential([
            iaa.Sometimes(self.aug_prob, iaa.imgcorruptlike.DefocusBlur(severity=(1,2))),
            iaa.Sometimes(self.aug_prob, iaa.imgcorruptlike.Contrast(severity=(1,2))),
            iaa.Sometimes(self.aug_prob, iaa.GammaContrast((0.5, 1.8))),
            iaa.Sometimes(self.aug_prob, iaa.AddToSaturation((-150, 10))),
            iaa.Sometimes(self.aug_prob, iaa.MotionBlur(k=(3,6))),
            iaa.Sometimes(self.aug_prob, iaa.AdditiveGaussianNoise(scale=(0, 0.07*255))),
            # iaa.Sometimes(self.aug_prob, iaa.UniformColorQuantizationToNBits(nb_bits=(3,7))),
            iaa.Sometimes(self.aug_prob, iaa.Grayscale(alpha=1.0)),
            iaa.Sometimes(self.aug_prob, iaa.JpegCompression(compression=(30, 70))),
        ])
        
        # generate metadata for the provided data path (requires a specific format)
        self.metadata = self._generate_metadata()

        # self.metadata[['scale_dict', 'res_dict', 'new_res_dict']] = self.metadata.apply(
        #     self._get_scale, axis = 1, result_type = 'expand')

        # balances datasets
        if self.balance_datasets:
            print('blancing datasets...') 
            self.metadata = self._balance_metadata(n_samples = self.n_samples_per_dataset)
            print(self.metadata.groupby('dataset').size())

        # self.metadata_path = os.path.join(data_path, 'posetail_metadata.csv')
        # self.metadata.to_csv(self.metadata_path, index = False)


    def __len__(self): 
        return len(self.metadata)


    def __getitem__(self, idx): 
        
        row = self.metadata.loc[idx].to_dict()
        print(type(row))
        start_ix = row['start_ix']
        # end_ix = row['end_ix']
        interval = row['interval']
        end_ix = start_ix + self.n_frames * interval 
        fnums = torch.arange(start_ix, end_ix, interval)

        # pprint(row)
        
        # load keypoints and visibilities (if present)
        data = np.load(row['pose_path'])
        coords = data['pose'][:, start_ix:end_ix:interval, :, :] 
        coords = torch.tensor(coords, dtype = torch.float32, device = 'cpu')

        vis = None
        if 'vis' in data: 
            vis = data['vis'][:, start_ix:end_ix:interval, :, :]
            vis = torch.tensor(vis, dtype = torch.float32, device = 'cpu')
            vis[torch.isnan(vis)] = 1
            vis = vis.bool()

        # only augment some of the samples
        should_augment = np.random.random() < 0.6
            
        # sample a random subject with 0.5 probability if using a 
        # multi-subject dataset
        if np.random.random() < 0.5:
            ix_sample = np.random.randint(coords.shape[0])
            coords = coords[ix_sample, None]
            if vis is not None:
                vis = vis[ix_sample, None]
            
        coords = rearrange(coords, 's t n r -> t (s n) r') # (time, n_kpts, 3)
        if vis is not None:
            vis = rearrange(vis, 's t n c -> t (s n) c') # (time, n_kpts, cams)
        
        # load camera resolutions for resizing
        # res_dict = json.loads(row['res_dict'])
        # new_res_dict = json.loads(row['new_res_dict'])
        # scale_dict = json.loads(row['scale_dict'])

        img_path = row['img_path']
        cam_names = get_dirs(img_path)
        img_fnames = sorted(os.listdir(os.path.join(img_path, cam_names[0])))[start_ix:end_ix:interval]

        # sample a number of camera views from a set of calibrated cameras
        if self.cams_to_sample: 
            
            if isinstance(self.cams_to_sample, int): 
                num_cams_to_sample = self.cams_to_sample
            else: # sample between a high and low bound
                num_cams_to_sample = np.random.randint(self.cams_to_sample[0], self.cams_to_sample[1] + 1)

            if len(cam_names) > num_cams_to_sample:
                ix_cams = np.random.choice(len(cam_names), size = num_cams_to_sample, replace = False)
                cam_names = [cam_names[i] for i in ix_cams]
                # determine visibilities only from the sampled cameras
                if vis is not None: 
                    vis = vis[:, :, ix_cams]


        if vis is not None:
            vis = vis.sum(dim = -1) >= self.cam_thresh_for_vis # (time, n_kpts)                
                
        # filter coords based on which coords are visible
        # in the first frame (will sample from these)
        if vis is not None:
            mask = vis[0].bool()
            coords = coords[:, mask, :]
            vis = vis[:, mask]

        # filter coords that are not nan throughout
        mask = torch.all(torch.isfinite(coords), dim=(0, 2))
        coords = coords[:, mask]

        if vis is not None: 
            vis = vis[:, mask]

        cgroup, offset_dict, cam_type = self._load_cameras(row['camera_metadata_path']) 
        cgroup = cgroup.subset_cameras_names(cam_names)
        cgroup = format_camera_group(cgroup, offset_dict, cam_type, device = 'cpu')
        
        # filter points that are visible in enough views
        if self.enable_kpt_filtering:

            s, n, _ = coords.shape
            coords_flat = rearrange(coords, 's n r -> (s n) r')
            all_visible = torch.stack([is_point_visible(cam, coords_flat) 
                                       for cam in cgroup])
            count_flat = torch.sum(all_visible, dim = 0)
            count = rearrange(count_flat, '(s n) -> s n', s = s, n = n)
            good = torch.all(count >= self.cam_thresh_for_vis, dim = 0)
            coords = coords[:, good, :]

            # filter vis if available
            if vis is not None:
                vis = vis[:, good]

        # sample a random number of keypoints from available tracks 
        if self.kpts_to_sample: 

            if isinstance(self.kpts_to_sample, int): 
                num_kpts_to_sample = self.kpts_to_sample
            else: # sample between a high and low bound 
                num_kpts_to_sample = np.random.randint(self.kpts_to_sample[0], self.kpts_to_sample[1] + 1)

            # sample if there are more keypoints than the number to sample
            if coords.shape[1] > num_kpts_to_sample:   
                ix_p = np.random.choice(coords.shape[1], size = num_kpts_to_sample, replace = False)
                coords = coords[:, ix_p]

                # sample corresponding visibilities
                if vis is not None: 
                    vis = vis[:, ix_p]

        # failed to sample coordinates, just get another random sample
        if coords.shape[1] < 1:
            return self.__getitem__(np.random.randint(self.__len__()))

        # compute total movement in pixels, averaged across cameras
        p2d = project_points_torch(cgroup, coords)
        movement = torch.linalg.norm(torch.diff(p2d, dim=1), dim=-1)
        total_movement = torch.mean(torch.sum(movement, dim=1), dim=0)
        # should have at least 12 pixels of movement over the sampled frames
        good = total_movement >= 12
        if torch.sum(good) < 2: # not enough points with movement
            return self.__getitem__(np.random.randint(self.__len__()))            
        
        # cropping around coordinates
        #   helps for small animals in large arenas
        if self.crop_to_points:
            # compute crops locations
            p2d = project_points_torch(cgroup, coords)
            crops = []
            for cnum in range(p2d.shape[0]):
                size = cgroup[cnum]['size']
                pflat = p2d[cnum].reshape(-1, 2)
                good = torch.all(torch.isfinite(pflat), dim=1)
                pflat = pflat[good]
                low = torch.clamp(torch.min(pflat, dim=0).values - 20, torch.tensor([0,0]), size).to(torch.int32)
                high = torch.clamp(torch.max(pflat, dim=0).values + 20, torch.tensor([0,0]), size).to(torch.int32)

                current_width = high[0] - low[0]
                current_height = high[1] - low[1]

                min_dim = max(self.min_crop_dim, current_width//2, current_height//2)

                if current_width < min_dim:
                    # Expand horizontally around center
                    center_x = (low[0] + high[0]) // 2
                    low[0] = torch.clamp(center_x - min_dim // 2, 0, size[0] - min_dim)
                    high[0] = torch.clamp(low[0] + min_dim, 0, size[0])
                    low[0] = high[0] - min_dim  # Adjust if clamping moved the window

                if current_height < min_dim:
                    # Expand vertically around center
                    center_y = (low[1] + high[1]) // 2
                    low[1] = torch.clamp(center_y - min_dim // 2, 0, size[1] - min_dim)
                    high[1] = torch.clamp(low[1] + min_dim, 0, size[1])
                    low[1] = high[1] - min_dim  # Adjust if clamping moved the window

                crops.append(torch.cat([low, high]))
            

            # camera crops
            camera_group_cropped = []
            for cnum in range(len(cgroup)):
                x1, y1, x2, y2 = crops[cnum]
                cam = dict(cgroup[cnum])
                cam['offset'] = cam['offset'] + torch.tensor([x1, y1], dtype=torch.int32, device='cpu')
                cam['size'] = torch.tensor([x2 - x1, y2 - y1], dtype=torch.int32, device='cpu')
                camera_group_cropped.append(cam)
            cgroup = camera_group_cropped

        # resize cameras
        if self.max_res != -1:
            target_res = np.random.randint(self.min_res, self.max_res + 1)
            camera_group_scaled = []
            for cnum in range(len(cgroup)):
                cam = dict(cgroup[cnum])
                name = cam['name']
                size = cam['size']
                scale = float(target_res) / max(size)
                cam['size'] = torch.round(size * scale).to(torch.int32)
                cam['mat'] = cam['mat'] * scale
                cam['mat'][2, 2] = 1
                if 'offset' in cam:
                    cam['offset'] = cam['offset'] * scale
                camera_group_scaled.append(cam)
            cgroup = camera_group_scaled
        
        views = []
        for cnum, cam_name in enumerate(cam_names):

            # we apply the same augmentation per camera
            # (thus assuming that each recording is at least self-consistent)
            aug_det = self.aug.to_deterministic()
            
            imgs = []

            if self.crop_to_points:
                x1, y1, x2, y2 = crops[cnum]
            
            # load images from paths and resize to desired resolution
            for img_fname in img_fnames: 

                cam_img_path = os.path.join(img_path, cam_name, img_fname)
                img = cv2.imread(cam_img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.crop_to_points:
                    img = img[y1:y2, x1:x2]
                
                if self.max_res != -1:
                    img = cv2.resize(img, dsize = cgroup[cnum]['size'].tolist())

                if should_augment:
                    img = aug_det(image=img)
                imgs.append(img)

            views.append(torch.tensor(np.array(imgs), dtype = torch.float32) / 255.0)

        # print(row['dataset'])

        return views, coords, vis, fnums, cgroup, row


    def _get_start_ixs(self, coords):

        if self.split == 'train': 
            start_ixs = self._get_start_ixs_train(coords)
        else: 
            start_ixs = self._get_start_ixs_test(coords)

        return start_ixs
    

    def _get_start_ixs_train(self, coords):

        start_ixs = []
        intervals = []

        for interval in [1, 2, 4]:
            for i in range(coords.shape[0] - self.n_frames * interval + 1): 
                
                start = i
                end = i + self.n_frames * interval
                coords_subset = coords[start:end:interval, :, :]        
                
                # if not all nans in the starting frame 
                if np.isfinite(coords_subset[0]).any():
                    start_ixs.append(i)
                    intervals.append(interval)
   
   
        start_ixs = np.array(start_ixs)
        intervals = np.array(intervals)
        
        return start_ixs, intervals

    def _get_start_ixs_test(self, coords):

        safe = 0
        start_ixs = []
        intervals = []
        
        n_start = coords.shape[0] // self.n_frames
        if coords.shape[0] % self.n_frames != 0: 
            n_start += 1
        
        for i in range(n_start): 

            if safe > 0:
                safe = safe - 1 
                continue

            coords_subset = coords[i:i + self.n_frames, :, :]
            enough_frames = coords_subset.shape[1] == self.n_frames
            
            # if not all nans in the starting frame and enough_frames: 
            if np.isfinite(coords_subset[0]).any() and enough_frames:
                start_ixs.append(i)
                intervals.append(1)
                safe = self.n_frames - 1

        start_ixs = np.array(start_ixs)
        intervals = np.array(intervals)
        
        return start_ixs, intervals


    def _generate_metadata(self, track_3d = True): 
            
        rows = []
        mode = '3d' # if track_3d else '2d' - not yet implemented

        for dataset in get_dirs(self.data_path):
            
            # NOTE: split folder structure must match here
            dataset_path = os.path.join(self.data_path, dataset, self.split_dir)

            # skip dataset if this particular split doesn't exist
            if not os.path.exists(dataset_path): 
                continue

            for session in get_dirs(dataset_path): 
                session_path = os.path.join(dataset_path, session)

                for trial in get_dirs(session_path):
                    # get paths to metadata, 3d pose, and images
                    trial_path = os.path.join(session_path, trial)

                    metadata_path = os.path.join(trial_path, 'metadata.yaml')
                    assert os.path.exists(metadata_path)
                    cam_metadata = load_yaml(metadata_path)

                    camera_height_dict = cam_metadata['camera_heights']
                    camera_width_dict = cam_metadata['camera_widths']

                    pose_path = os.path.join(trial_path, f'pose{mode}.npz')
                    assert os.path.exists(pose_path)

                    img_path = os.path.join(trial_path, 'img')
                    cams = os.listdir(img_path)
                    assert len(cams) > 0

                    # get starting indices 
                    data = np.load(pose_path)
                    coords = data['pose']

                    coords = rearrange(coords, 's t n r -> t (s n) r') # (time, n_kpts, 3)
                    start_ixs, intervals = self._get_start_ixs(coords)

                    # n_batches = len(imgs) // self.n_frames
                    # start_ixs = np.arange(0, len(imgs), self.n_frames)[:n_batches]
                    # end_ixs = start_ixs + self.n_frames

                    # add a row to the metadata that will correspond
                    # to each sample within a batch
                    for start_ix, interval in zip(start_ixs, intervals): 
                        row = [dataset, session, trial, metadata_path,
                               pose_path, img_path, start_ix, interval, 
                               camera_height_dict, camera_width_dict]
                        rows.append(row)

        columns = ['dataset', 'session', 'trial', 'camera_metadata_path', 
                   'pose_path', 'img_path', 'start_ix', 'interval', 
                   'camera_heights', 'camera_widths']

        df = pd.DataFrame(rows, columns = columns)
        df['camera_heights'] = df['camera_heights'].apply(json.dumps)
        df['camera_widths'] = df['camera_widths'].apply(json.dumps)

        return df 
    
    def _balance_group(self, df, n_samples = 1000, random_state = 3): 

        duplicates = int(np.ceil(n_samples / len(df)))

        if duplicates > 1: 
            df = pd.concat([df] * duplicates, axis = 0)# .reset_index(drop = True)

        df_balanced = df.sample(n = n_samples, random_state = random_state)

        return df_balanced

    def _balance_metadata(self, n_samples = -1, random_state = 3): 

        self.metadata['dataset2'] = self.metadata['dataset'].copy()

        # balance the dataset according to the dataset with the most samples
        if n_samples == -1: 
            n_samples = self.metadata.groupby('dataset2').size().max()

        # balance and sample based on a predefined number of samples
        df_balanced = (self.metadata.groupby('dataset2')
                           .apply(lambda x: self._balance_group(x, 
                                                                n_samples = n_samples, 
                                                                random_state = random_state), 
                                                                include_groups = False)
                           .reset_index(drop = True))
        
        return df_balanced

    def _get_scale(self, row): 

        scale_dict = {}
        res_dict = {}
        new_res_dict = {}

        camera_height_dict = json.loads(row['camera_heights'])
        camera_width_dict = json.loads(row['camera_widths'])

        for cam_name, height in camera_height_dict.items():

            width = camera_width_dict[cam_name]

            if self.max_res != -1: 
                scale = self.max_res / max(height, width)
            else: 
                scale = 1

            orig_res = [width, height]
            new_res = [round(width * scale), round(height * scale)]
            # xy_scale = (orig_res[0] / new_res[0], orig_res[1] / new_res[1])

            scale_dict[cam_name] = scale
            res_dict[cam_name] = orig_res
            new_res_dict[cam_name] = new_res
        
        scale_dict = json.dumps(scale_dict)
        res_dict = json.dumps(res_dict)
        new_res_dict = json.dumps(new_res_dict)

        return scale_dict, res_dict, new_res_dict


    def _load_cameras(self, camera_metadata_path):

        cam_metadata = load_yaml(camera_metadata_path)
        offset_dict = None
        cam_type = 'pinhole'

        intrinsics_dict = cam_metadata['intrinsic_matrices']
        extrinsics_dict = cam_metadata['extrinsic_matrices']
        distortions_dict = cam_metadata['distortion_matrices']
        heights_dict = cam_metadata['camera_heights']
        widths_dict = cam_metadata['camera_widths']

        if 'offset_dict' in cam_metadata: 
            offset_dict = cam_metadata['offset_dict']

        if 'cam_type' in cam_metadata: 
            cam_type = cam_metadata['cam_type']

        # sort camera names either numerically or alphabetically
        cam_names = list(intrinsics_dict.keys())

        if all(cam_name.isdigit() for cam_name in cam_names):
            cam_names = sorted(cam_names, key = int)
        else: 
            cam_names = sorted(cam_names) 

        cams = []

        for cam_name in cam_names: 

            rvec, tvec = disassemble_extrinsics(extrinsics_dict[cam_name])

            cam = Camera(
                matrix = intrinsics_dict[cam_name],
                dist = distortions_dict[cam_name],
                rvec = rvec,
                tvec = tvec,
                name = cam_name)

            width = widths_dict[cam_name]
            height = heights_dict[cam_name]
            cam.set_size((width, height))
            # cam.resize_camera(scale_dict[cam_name])
            cams.append(cam)

            # if offset_dict: 
            #     offsets = offset_dict[cam_name]
            #     offset_dict[cam_name] = [offsets[0] * scale_dict[cam_name], offsets[1] * scale_dict[cam_name]]

        cgroup = CameraGroup(cams)

        return cgroup, offset_dict, cam_type
