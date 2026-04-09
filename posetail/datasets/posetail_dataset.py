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

from concurrent.futures import ThreadPoolExecutor

def load_image(cam_img_path, crop_coords=None, target_size=None):
    img = cv2.imread(cam_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if crop_coords is not None:
        x1, y1, x2, y2 = crop_coords
        img = img[y1:y2, x1:x2]
    
    if target_size is not None:
        img = cv2.resize(img, target_size)
            
    return img

def get_rows_trial(trial_path, n_frames, split, context,
                   mode='3d'):
    dataset, session, trial = context
    
    metadata_path = os.path.join(trial_path, 'metadata.yaml')
    assert os.path.exists(metadata_path)
    # cam_metadata = load_yaml(metadata_path)
    # camera_height_dict = cam_metadata['camera_heights']
    # camera_width_dict = cam_metadata['camera_widths']

    img_path = os.path.join(trial_path, 'img')
    cams = os.listdir(img_path)
    assert len(cams) > 0

    pose_path = os.path.join(trial_path, f'pose{mode}.npz')
    assert os.path.exists(pose_path)

    # get starting indices 
    data = np.load(pose_path)
    coords = torch.as_tensor(data['pose'])

    coords = rearrange(coords, 's t n r -> t (s n) r') # (time, n_kpts, 3)
    start_ixs, intervals = get_start_ixs(coords, n_frames, split)

    # n_batches = len(imgs) // self.n_frames
    # start_ixs = np.arange(0, len(imgs), self.n_frames)[:n_batches]
    # end_ixs = start_ixs + self.n_frames

    # add a row to the metadata that will correspond
    # to each sample within a batch
    rows = []
    for start_ix, interval in zip(start_ixs, intervals): 
        row = [dataset, session, trial, metadata_path,
               pose_path, img_path, start_ix, interval, 
               # camera_height_dict, camera_width_dict
               ]
        rows.append(row)

    return rows


def get_start_ixs(coords, n_frames, split):

    if split == 'train': 
        start_ixs = get_start_ixs_train(coords, n_frames)
    else: 
        start_ixs = get_start_ixs_test(coords, n_frames)

    return start_ixs


def get_start_ixs_train(coords, n_frames):

    start_ixs = []
    intervals = []

    for interval in [1, 2, 4]:
        for i in range(coords.shape[0] - n_frames * interval + 1): 

            start = i
            end = i + n_frames * interval
            coords_subset = coords[start:end:interval, :, :]        

            # if not all nans in the starting frame 
            if np.isfinite(coords_subset[0]).any():
                start_ixs.append(i)
                intervals.append(interval)


    start_ixs = np.array(start_ixs)
    intervals = np.array(intervals)

    return start_ixs, intervals

def get_start_ixs_test(coords, n_frames):

    safe = 0
    start_ixs = []
    intervals = []

    for i in range(coords.shape[0]): 

        if safe > 0:
            safe = safe - 1 
            continue

        coords_subset = coords[i:i + n_frames, :, :]
        enough_frames = coords_subset.shape[0] == n_frames

        # if not all nans in the starting frame and enough_frames: 
        if np.isfinite(coords_subset[0]).any() and enough_frames:
            start_ixs.append(i)
            intervals.append(1)
            safe = n_frames - 1

    start_ixs = np.array(start_ixs)
    intervals = np.array(intervals)

    return start_ixs, intervals

    
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

    # # mask nan coordinates in the first frame
    coords = torch.stack(batch[1], axis = 0)
    # mask = torch.isfinite(coords).all(dim = -1).all(dim = 1).all(dim = 0)
    # coords_masked = coords[:, :, mask, :]

    # # get corresponding visibilities if present
    # vis_masked = None
    # if batch[2][0] is not None: 
    #     vis = torch.stack(batch[2], axis = 0)
    #     vis_masked = vis[:, :, mask].unsqueeze(-1)

    vis = None
    if batch[2][0] is not None: 
        vis = torch.stack(batch[2], axis = 0)[..., None]

    vis_2d = None
    if batch[7][0] is not None:
        vis_2d = torch.stack(batch[7], axis=0)[..., None]

        
    rows = batch[5][0]
    query_times = torch.stack(batch[6])
    
    batch = edict({'views': views, 
                   'coords': coords,
                   'query_times': query_times,
                   'vis': vis,
                   'vis_2d': vis_2d,
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
        self.datasets_to_exclude = config.dataset.get('datasets_to_exclude', [])
        self.n_frames = config.dataset[split].get('n_frames', 16)
        self.max_res = config.dataset[split].get('max_res', -1) # -1 means no resizing
        self.min_res = config.dataset[split].get('min_res', self.max_res) # only used when max_res != -1
        self.aug_prob = config.dataset[split].get('aug_prob', 0.25)

        self.crop_to_points = config.dataset[split].get('crop_to_points', True)
        self.min_crop_dim = config.dataset[split].get('min_crop_dim', 64)

        # for sampling cameras, keypoints
        self.cams_to_sample = format_sample_input(config.dataset[split].get('cams_to_sample', None))
        self.kpts_to_sample = format_sample_input(config.dataset[split].get('kpts_to_sample', None))
        self.speed_thresh = config.dataset[split].get('speed_thresh', None) 
        self.prop_dynamic_kpts_to_sample = config.dataset[split].get('prop_dynamic_kpts_to_sample', 0.7)
        self.cam_thresh_for_vis = config.dataset[split].get('cam_thresh_for_vis', 1) 
        self.enable_kpt_filtering = config.dataset[split].get('enable_kpt_filtering', False)
        self.query_anytime = config.dataset[split].get('query_anytime', False)
        self.query_edge_bias = config.dataset[split].get('query_edge_bias', 3.0)
        
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

        print("total length:", len(self.metadata))
        self.good_index = np.ones(len(self.metadata), dtype='bool')
        # self.metadata_path = os.path.join(data_path, 'posetail_metadata.csv')
        # self.metadata.to_csv(self.metadata_path, index = False)


    def __len__(self): 
        return len(self.metadata)


    def __getitem__(self, idx):
        start = idx
        out = None
        while True:
            if self.good_index[start]:
                out = self.get_item_actual(start)
            if out is not None:
                return out
            
            self.good_index[start] = False
            start = np.random.randint(len(self.metadata))
            if np.sum(self.good_index) == 0:
                return None # no valid samples
            
            # if start >= self.__len__():
            #     start = np.random.randint(self.__len__())
                
        
    def get_item_actual(self, idx):
        # print(idx, "started")
        
        row = self.metadata.loc[idx].to_dict()
        start_ix = row['start_ix']
        # end_ix = row['end_ix']
        interval = row['interval']
        end_ix = start_ix + self.n_frames * interval 
        fnums = torch.arange(start_ix, end_ix, interval)
        
        # load keypoints 
        data = np.load(row['pose_path'])
        coords = data['pose'][:, start_ix:end_ix:interval, :, :] 
        coords = torch.tensor(coords, dtype = torch.float32, device = 'cpu')

        # load visibilities (if present)
        vis = None
        vis_2d = None
        if 'vis' in data: 
            vis = data['vis'][:, start_ix:end_ix:interval, :, :]
            vis = torch.tensor(vis, dtype = torch.float32, device = 'cpu')
            vis_2d = vis.clone()
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
                vis_2d = vis_2d[ix_sample, None]
            
        coords = rearrange(coords, 's t n r -> t (s n) r') # (time, n_kpts, 3)
        if vis is not None:
            vis = rearrange(vis, 's t n c -> t (s n) c') # (time, n_kpts, cams)
            vis_2d = rearrange(vis_2d, 's t n c -> t (s n) c')
        
        # load camera resolutions for resizing
        # res_dict = json.loads(row['res_dict'])
        # new_res_dict = json.loads(row['new_res_dict'])
        # scale_dict = json.loads(row['scale_dict'])

        img_path = row['img_path']
        cam_names = get_dirs(img_path)
        img_fnames = sorted(os.listdir(os.path.join(img_path, cam_names[0])))[start_ix:end_ix:interval]

        # sample a number of camera views from a set of calibrated cameras
        if self.cams_to_sample: 
            coords, vis, vis_2d, cam_names = self.sample_cameras(coords, vis, vis_2d, cam_names)

        if vis is not None:
            vis = vis.sum(dim = -1) >= self.cam_thresh_for_vis # (time, n_kpts)                

        # filter coords based on which coords are visible enough times
        if self.query_anytime:
            valid_mask = torch.isfinite(coords[..., 0])  # (time, n_kpts)
            if vis is not None:
                valid_mask = valid_mask & vis
            sum_good = torch.sum(valid_mask, dim=0) 

            # some number visible and not nan 
            mask = sum_good >= 2
            coords = coords[:, mask]
            if vis is not None: 
                vis = vis[:, mask]
                vis_2d = vis_2d[:, mask]
        else:
            # filter in the first frame (will sample from these)
            if vis is not None:
                mask = vis[0].bool()
                coords = coords[:, mask, :]
                vis = vis[:, mask]
                vis_2d = vis_2d[:, mask]

            # filter coords that are not nan throughout
            mask = torch.all(torch.isfinite(coords), dim=(0, 2))
            coords = coords[:, mask]
            if vis is not None: 
                vis = vis[:, mask]
                vis_2d = vis_2d[:, mask]
            
        # load cameras
        cgroup, offset_dict, cam_type = self._load_cameras(row['camera_metadata_path']) 
        cgroup = cgroup.subset_cameras_names(cam_names)
        cgroup = format_camera_group(cgroup, offset_dict, cam_type, device = 'cpu')
        
        # filter points that are visible in enough views
        if self.enable_kpt_filtering:
            coords, vis, vis_2d = self.filter_keypoints(coords, vis, vis_2d, cgroup)

        if coords.shape[1] < 2:
            return None
                
        # compute total movement and speed in pixels, averaged across cameras
        p2d = project_points_torch(cgroup, coords)
        movement = torch.linalg.norm(torch.diff(p2d, dim = 1), dim = -1)
        movement = torch.nan_to_num(movement, 0.0)
        total_movement = torch.mean(torch.sum(movement, dim = 1), dim = 0)
        avg_speed = torch.mean(torch.mean(movement, dim = 1), dim = 0) 

        # should have at least 12 pixels of movement over the sampled frames
        good = total_movement >= 12
        if torch.sum(good) < 2: # not enough points with movement
            return None

        # sample a random number of keypoints from available tracks 
        if self.kpts_to_sample: 
            coords, vis, vis_2d = self.sample_keypoints(coords, vis, vis_2d, total_movement, avg_speed)

        # failed to sample coordinates, just get another random sample
        if coords.shape[1] < 2:
            return None
        
        # cropping around coordinates
        # helps for small animals in large arenas
        if self.crop_to_points:
            cgroup, crops = self.crop_cgroup_to_points(cgroup, coords)

        # resize cameras
        if self.max_res != -1:
            cgroup = self.resize_camera_group(cgroup)

        # arbitrary camera rotation
        cgroup, coords = self.rotate_camera_group(cgroup, coords)

        # setup possible query_times (n_kpts)
        if self.query_anytime:
            query_times = []
            for kpt_idx in range(coords.shape[1]):
                good = torch.isfinite(coords[:, kpt_idx, 0])
                if vis is not None:
                    good = good & vis[:, kpt_idx]
                valid_times = torch.where(good)[0]
                query_time = self.sample_query_time(valid_times)
                query_times.append(query_time.item())
            query_times = torch.tensor(query_times, dtype=torch.int32, device='cpu')            
        else:
            query_times = torch.zeros((coords.shape[1],), dtype=torch.int32, device='cpu')
        
        # apply augmentation
        with ThreadPoolExecutor(max_workers=24) as executor:
            views_unloaded = []
            for cnum, cam_name in enumerate(cam_names):

                # we apply the same augmentation per camera
                # (thus assuming that each recording is at least self-consistent)
                #
                if self.max_res != -1:
                    target_size = cgroup[cnum]['size'].tolist()
                else:
                    target_size = None

                if self.crop_to_points:
                    crop_coords = crops[cnum]
                else:
                    crop_coords = None

                futures = []
                # load images from paths and resize to desired resolution
                for img_fname in img_fnames: 
                    cam_img_path = os.path.join(img_path, cam_name, img_fname)
                    future = executor.submit(
                        load_image,
                        cam_img_path, crop_coords, target_size)
                    futures.append(future)
                views_unloaded.append(futures)

            views = []
            for futures in views_unloaded:
                imgs = [f.result() for f in futures]
                if should_augment:
                    aug_det = self.aug.to_deterministic()
                    imgs = [aug_det(image=img) for img in imgs]

                imgs = torch.tensor(np.array(imgs), dtype = torch.float32, device='cpu')
                imgs = imgs / 255.0
                views.append(imgs)

        return views, coords, vis, fnums, cgroup, row, query_times, vis_2d


    def crop_cgroup_to_points(self, cgroup, coords): 
            
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

            min_dim = max(self.min_crop_dim, current_width, current_height)

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
        
        return camera_group_cropped, crops


    def filter_keypoints(self, coords, vis, vis_2d, cgroup): 

        # filter keypoints that are not visible from enough views 
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
            vis_2d = vis_2d[:, good]

        return coords, vis, vis_2d


    def sample_cameras(self, coords, vis, vis_2d, cam_names): 

        # sample a number of camera views from a set of calibrated cameras
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
                vis_2d = vis_2d[:, :, ix_cams]

        return coords, vis, vis_2d, cam_names


    def sample_keypoints(self, coords, vis, vis_2d, total_movement, avg_speed): 

        if isinstance(self.kpts_to_sample, int): 
            num_kpts_to_sample = self.kpts_to_sample

        else: # sample between a high and low bound 
            num_kpts_to_sample = np.random.randint(self.kpts_to_sample[0], self.kpts_to_sample[1] + 1)

        # sample if there are more keypoints than the number to sample
        if coords.shape[1] > num_kpts_to_sample:
            # sample a proportion of static vs dynamic points if a speed thresh is provided
            if self.speed_thresh is not None: 

                dynamic_mask = avg_speed >= self.speed_thresh
                static_mask = ~dynamic_mask

                num_dynamic = int(num_kpts_to_sample * self.prop_dynamic_kpts_to_sample)
                num_static = num_kpts_to_sample - num_dynamic

                dynamic_idx = torch.where(dynamic_mask)[0].cpu().numpy()
                static_idx = torch.where(static_mask)[0].cpu().numpy()

                num_dynamic = min(num_dynamic, len(dynamic_idx))
                num_static = min(num_static, len(static_idx))

                sampled_dynamic = np.random.choice(dynamic_idx, size = num_dynamic, replace = False) if len(dynamic_idx) > 0 else []
                sampled_static = np.random.choice(static_idx, size = num_static, replace = False) if len(static_idx) > 0 else []

                ix_p = np.concatenate([sampled_dynamic, sampled_static])
                np.random.shuffle(ix_p)
                coords = coords[:, ix_p]

            # otherwise, default to sampling probabilities based on total movement
            else: 
                prob = (total_movement + 2) / torch.sum(total_movement + 2)
                prob = prob.numpy()
                
                ix_p = np.random.choice(coords.shape[1], size = num_kpts_to_sample,
                                        replace = False, p = prob)
                coords = coords[:, ix_p]

            # sample corresponding visibilities
            if vis is not None: 
                vis = vis[:, ix_p]
                vis_2d = vis_2d[:, ix_p]

        return coords, vis, vis_2d


    def sample_query_time(self, valid_times):
        valid_times = valid_times.to(torch.long)

        if len(valid_times) == 1:
            return valid_times[0].to(torch.int32)

        dist_to_start = valid_times
        dist_to_end = (self.n_frames - 1) - valid_times
        dist_to_edge = torch.minimum(dist_to_start, dist_to_end).to(torch.float32)

        weights = 1.0 / (dist_to_edge + 1.0)
        weights[valid_times == 0] *= self.query_edge_bias
        # weights[valid_times == (self.n_frames - 1)] *= self.query_edge_bias

        probs = weights / weights.sum()
        sample_ix = torch.multinomial(probs, 1)

        return valid_times[sample_ix].squeeze(0).to(torch.int32)


    def resize_camera_group(self, cgroup): 

        target_res = np.random.randint(self.min_res, self.max_res + 1)
        camera_group_scaled = []

        for cnum in range(len(cgroup)):

            cam = dict(cgroup[cnum])
            size = cam['size']
            scale = float(target_res) / max(size)
            cam['size'] = torch.round(size * scale).to(torch.int32)
            cam['mat'] = cam['mat'] * scale
            cam['mat'][2, 2] = 1

            if 'offset' in cam:
                cam['offset'] = cam['offset'] * scale

            camera_group_scaled.append(cam)

        return camera_group_scaled


    def rotate_camera_group(self, cgroup, coords): 
                
        rvec = np.random.uniform(-2*np.pi, 2*np.pi, size=3)
        rotmat, _ = cv2.Rodrigues(np.array(rvec))
        rotmat = torch.as_tensor(rotmat, device=coords.device, dtype=coords.dtype)
        coords = torch.matmul(coords, rotmat)

        rmat = torch.eye(4, device=coords.device, dtype=coords.dtype)
        rmat[:3,:3] = rotmat
        camera_group_rotated = list()

        for cam in cgroup:
            cam_rot = dict(cam)
            cam_rot['ext'] = torch.matmul(cam['ext'], rmat)
            cam_rot['ext_inv'] = torch.linalg.inv(cam_rot['ext'])
            camera_group_rotated.append(cam_rot)

        cgroup = camera_group_rotated 

        return cgroup, coords


    def _generate_metadata(self, track_3d = True): 
            
        rows = []
        mode = '3d' # if track_3d else '2d' - not yet implemented

        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = []
            for dataset in get_dirs(self.data_path):

                if dataset in self.datasets_to_exclude: 
                    continue

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
                        future = executor.submit(
                            get_rows_trial,
                            trial_path, self.n_frames, self.split,
                            (dataset, session, trial),
                            mode)
                        futures.append(future)
                        
            for future in futures:
                add_rows = future.result()
                rows.extend(add_rows)

        columns = ['dataset', 'session', 'trial', 'camera_metadata_path', 
                   'pose_path', 'img_path', 'start_ix', 'interval', 
                   # 'camera_heights', 'camera_widths'
                   ]

        df = pd.DataFrame(rows, columns = columns)
        # df['camera_heights'] = df['camera_heights'].apply(json.dumps)
        # df['camera_widths'] = df['camera_widths'].apply(json.dumps)

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

    # def _get_scale(self, row): 

    #     scale_dict = {}
    #     res_dict = {}
    #     new_res_dict = {}

    #     camera_height_dict = json.loads(row['camera_heights'])
    #     camera_width_dict = json.loads(row['camera_widths'])

    #     for cam_name, height in camera_height_dict.items():

    #         width = camera_width_dict[cam_name]

    #         if self.max_res != -1: 
    #             scale = self.max_res / max(height, width)
    #         else: 
    #             scale = 1

    #         orig_res = [width, height]
    #         new_res = [round(width * scale), round(height * scale)]
    #         # xy_scale = (orig_res[0] / new_res[0], orig_res[1] / new_res[1])

    #         scale_dict[cam_name] = scale
    #         res_dict[cam_name] = orig_res
    #         new_res_dict[cam_name] = new_res
        
    #     scale_dict = json.dumps(scale_dict)
    #     res_dict = json.dumps(res_dict)
    #     new_res_dict = json.dumps(new_res_dict)

    #     return scale_dict, res_dict, new_res_dict


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
