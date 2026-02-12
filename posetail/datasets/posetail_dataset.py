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

from posetail.datasets.utils import get_dirs, load_yaml, disassemble_extrinsics
from posetail.posetail.cube import project_points_torch, is_point_visible
from train_utils import format_camera_group, dict_to_device

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

    batch = edict({'views': views, 
                   'coords': coords_masked,
                   'vis': vis_masked,
                   'fnums': fnums, 
                   'cgroup': cgroup})

    return batch


def format_sample_input(x):

    if isinstance(x, int): 
        return x
    elif isinstance(x, list): 
        return tuple(x) 
    else: 
        return None


class PosetailDataset(Dataset): 

    def __init__(self, config, split, n_frames = 16, 
                 cam_thresh_for_vis = 1, enable_kpt_filtering = False, 
                 aug_prob = 0.25): 

        self.split = split
        assert split in {'train', 'val', 'test'}
        self.split_dir = config.dataset[split].get('split_dir')

        self.data_path = config.dataset.prefix
        self.n_frames = config.dataset[split].get('n_frames', n_frames)
        self.max_res = config.dataset[split].get('max_res', -1) # -1 means no resizing
        self.aug_prob = config.dataset[split].get('aug_prob', aug_prob)

        # for sampling cameras and keypoints 
        self.cams_to_sample = format_sample_input(config.dataset[split].get('cams_to_sample', None))
        self.kpts_to_sample = format_sample_input(config.dataset[split].get('kpts_to_sample', None))
        self.cam_thresh_for_vis = config.dataset[split].get('cam_thresh_for_vis', cam_thresh_for_vis) 
        self.enable_kpt_filtering = config.dataset[split].get('enable_kpt_filtering', enable_kpt_filtering)

        # augmentation
        self.aug = iaa.Sequential([
            iaa.Sometimes(aug_prob, iaa.imgcorruptlike.DefocusBlur(severity=(1,2))),
            iaa.Sometimes(aug_prob, iaa.imgcorruptlike.Contrast(severity=(1,2))),
            iaa.Sometimes(aug_prob, iaa.GammaContrast((0.5, 1.8))),
            iaa.Sometimes(aug_prob, iaa.AddToSaturation((-150, 10))),
            iaa.Sometimes(aug_prob, iaa.MotionBlur(k=(3,6))),
            iaa.Sometimes(aug_prob, iaa.AdditiveGaussianNoise(scale=(0, 0.08*255))),
            iaa.Sometimes(aug_prob, iaa.UniformColorQuantizationToNBits(nb_bits=(3,7))),
            iaa.Sometimes(aug_prob, iaa.Grayscale(alpha=1.0)),
            iaa.Sometimes(aug_prob, iaa.JpegCompression(compression=(30, 80))),
        ])
        
        # generate metadata for the provided data path (requires a specific format)
        self.metadata = self._generate_metadata()

        self.metadata[['scale_dict', 'res_dict', 'new_res_dict']] = self.metadata.apply(
            self._get_scale, axis = 1, result_type = 'expand')

        # self.metadata_path = os.path.join(data_path, 'posetail_metadata.csv')
        # self.metadata.to_csv(self.metadata_path, index = False)


    def __len__(self): 
        return len(self.metadata)


    def __getitem__(self, idx): 
        
        row = self.metadata.loc[idx].to_dict()
        start_ix = row['start_ix']
        end_ix = row['end_ix']
        fnums = torch.arange(start_ix, end_ix)

        # load keypoints and visibilities (if present)
        data = np.load(row['pose_path'])
        coords = data['pose'][:, start_ix:end_ix, :, :] 
        coords = torch.tensor(coords, dtype = torch.float32, device = 'cpu')
        coords = rearrange(coords, 's t n r -> t (s n) r') # (time, n_kpts, 3)

        vis = None
        if 'vis' in data: 
            vis = data['vis'][:, start_ix:end_ix, :, :]
            vis = torch.tensor(vis, dtype = torch.float32, device = 'cpu')
            vis = rearrange(vis, 's t n c -> t (s n) c') # (time, n_kpts, cams)

        # load camera resolutions for resizing
        res_dict = json.loads(row['res_dict'])
        new_res_dict = json.loads(row['new_res_dict'])
        scale_dict = json.loads(row['scale_dict'])

        img_path = row['img_path']
        cam_names = get_dirs(img_path)
        img_fnames = sorted(os.listdir(os.path.join(img_path, cam_names[0])))[start_ix:end_ix]
        views = []

        # sample a number of camera views from a set of calibrated cameras
        if self.cams_to_sample: 
            
            if isinstance(self.cams_to_sample, int): 
                num_cams_to_sample = self.cams_to_sample
            else: # sample between a high and low bound
                num_cams_to_sample = np.random.randint(self.cams_to_sample[0], self.cams_to_sample[1])

            if len(cam_names) > num_cams_to_sample:
                ix_cams = np.random.choice(len(cam_names), size = num_cams_to_sample, replace = False)
                cam_names = [cam_names[i] for i in ix_cams]

            # determine visibilities only from the sampled cameras
            if vis is not None: 
                vis = vis[:, :, ix_cams].sum(dim = -1) >= self.cam_thresh_for_vis # (time, n_kpts)                
                
        # filter coords based on which coords are visible
        # in the first frame (will sample from these)
        if vis is not None: 
            mask = vis[0].bool()
            coords = coords[:, mask, :].squeeze()
            vis = vis[:, mask].squeeze()

        # create camera group from camera parameters
        if len(cam_names) == 1: 
            cgroup = None

        else: 
            cgroup, offset_dict, cam_type = self._load_cameras(row['camera_metadata_path'], res_dict, scale_dict) 
            cgroup = cgroup.subset_cameras_names(cam_names)
            cgroup = format_camera_group(cgroup, offset_dict, cam_type, device = 'cpu')
            
            # filter points that are visible from at least 2 views
            if self.enable_kpt_filtering:

                s, n, _ = coords.shape
                coords_flat = rearrange(coords, 's n r -> (s n) r')
                all_visible = torch.stack([is_point_visible(cam, coords_flat) 
                                           for cam in cgroup])
                count_flat = torch.sum(all_visible, dim = 0)
                count = rearrange(count_flat, '(s n) -> s n', s = s, n = n)
                good = torch.all(count >= 2, dim = 0)
                coords = coords[:, good, :]

                # filter vis if available
                if vis is not None: 
                    vis = vis[:, good]

        # sample a random number of keypoints from available tracks 
        if self.kpts_to_sample: 

            if isinstance(self.kpts_to_sample, int): 
                num_kpts_to_sample = self.kpts_to_sample
            else: # sample between a high and low bound 
                num_kpts_to_sample = np.random.randint(self.kpts_to_sample[0], self.kpts_to_sample[1])

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
                
        for cam_name in cam_names:

            # we apply the same augmentation per camera
            # (thus assuming that each recording is at least self-consistent)
            aug_det = self.aug.to_deterministic()
            imgs = []
            
            # load images from paths and resize to desired resolution
            for img_fname in img_fnames: 

                cam_img_path = os.path.join(img_path, cam_name, img_fname)
                img = cv2.imread(cam_img_path)

                if self.max_res != 1: 
                    img = cv2.resize(img, dsize = new_res_dict[cam_name])

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = aug_det(image=img)
                imgs.append(img)

            views.append(torch.tensor(np.array(imgs), dtype = torch.float32) / 255.0)
    
        return views, coords, vis, fnums, cgroup


    def _get_start_ixs(self, coords):

        if self.split == 'train': 
            start_ixs = self._get_start_ixs_train(coords)
        else: 
            start_ixs = self._get_start_ixs_test(coords)

        return start_ixs
    

    def _get_start_ixs_train(self, coords):

        start_ixs = []

        for i in range(coords.shape[1] - self.n_frames + 1): 

            coords_subset = coords[:, i:i + self.n_frames, :, :]        
            mask = np.isfinite(coords_subset)
            visible_coords = mask.all(axis = -1).all(axis = 1).squeeze(0)

            # if not all nans in the starting frame: 
            if np.sum(visible_coords) > 0:
                start_ixs.append(i)

        start_ixs = np.array(start_ixs)

        return start_ixs

    def _get_start_ixs_test(self, coords):

        safe = 0
        start_ixs = []

        for i in range(coords.shape[1]): 

            if safe > 0:
                safe = safe - 1 
                continue

            coords_subset = coords[:, i:i + self.n_frames, :, :]
            enough_frames = coords_subset.shape[1] == self.n_frames
            
            mask = np.isfinite(coords_subset)
            visible_coords = mask.all(axis = -1).all(axis = 1).squeeze(0)

            # if not all nans in the starting frame and enough_frames: 
            if np.sum(visible_coords) > 0 and enough_frames:
                start_ixs.append(i)
                safe = self.n_frames - 1

        start_ixs = np.array(start_ixs)

        return start_ixs


    def _generate_metadata(self, track_3d = True): 
            
        rows = []
        mode = '3d' # if track_3d else '2d' - not yet implemented

        for dataset in get_dirs(self.data_path): 
            
            # NOTE: split folder structure must match here
            dataset_path = os.path.join(self.data_path, dataset, self.split_dir)

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
                    start_ixs = self._get_start_ixs(coords)

                    # n_batches = len(imgs) // self.n_frames
                    # start_ixs = np.arange(0, len(imgs), self.n_frames)[:n_batches]
                    end_ixs = start_ixs + self.n_frames

                    # add a row to the metadata that will correspond
                    # to each sample within a batch
                    for start_ix, end_ix in zip(start_ixs, end_ixs): 
                        row = [dataset, session, trial, metadata_path,
                               pose_path, img_path, start_ix, end_ix, 
                               camera_height_dict, camera_width_dict]
                        rows.append(row)

        columns = ['dataset', 'session', 'trial', 'camera_metadata_path', 
                   'pose_path', 'img_path', 'start_ix', 'end_ix', 
                   'camera_heights', 'camera_widths']

        df = pd.DataFrame(rows, columns = columns)
        df['camera_heights'] = df['camera_heights'].apply(json.dumps)
        df['camera_widths'] = df['camera_widths'].apply(json.dumps)

        return df 

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


    def _load_cameras(self, camera_metadata_path, res_dict, scale_dict):

        cam_metadata = load_yaml(camera_metadata_path)
        offset_dict = None
        cam_type = 'pinhole'

        intrinsics_dict = cam_metadata['intrinsic_matrices']
        extrinsics_dict = cam_metadata['extrinsic_matrices']
        distortions_dict = cam_metadata['distortion_matrices']

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

            cam.set_size(res_dict[cam_name])
            cam.resize_camera(scale_dict[cam_name])
            cams.append(cam)

            if offset_dict: 
                offsets = offset_dict[cam_name]
                offset_dict[cam_name] = [offsets[0] * scale_dict[cam_name], offsets[1] * scale_dict[cam_name]]

        cgroup = CameraGroup(cams)

        return cgroup, offset_dict, cam_type
