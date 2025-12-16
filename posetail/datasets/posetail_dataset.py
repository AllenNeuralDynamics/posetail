import os 
import cv2
import json

import torch 
from torch.utils.data import Dataset

import numpy as np
import pandas as pd 

from aniposelib.cameras import CameraGroup, Camera
from easydict import EasyDict as edict

from posetail.datasets.utils import get_dirs, load_yaml, disassemble_extrinsics
from posetail.posetail.cube import project_points_torch
from einops import rearrange

from train_utils_lightning import format_camera_group

def custom_collate(batch):
    ''' 
    custom collate functon to enable returning 
    non-tensor, non-list, etc type objects from 
    the default collate function
    '''
    batch = list(zip(*batch))

    views = [torch.stack(v, dim = 0) for v in zip(*list(batch[0]))]
    coords = torch.stack(batch[1], axis = 0).squeeze(1)
    mask = torch.isfinite(coords).all(dim = -1).all(dim = 1).all(dim = 0)
    coords_masked = coords[:, :, mask, :]
    fnums = torch.stack(batch[2], axis = 0)

    if len(batch) == 3:
        batch = edict({'views': views, 
                       'coords': coords_masked,
                       'fnums': fnums})
    else: 
        cgroup = batch[3][0]
        batch = edict({'views': views, 
                       'coords': coords_masked,
                       'fnums': fnums, 
                       'cgroup': cgroup})

    return batch


class PosetailDataset(Dataset): 

    def __init__(self, data_path, track_3d = True, n_frames = 16, max_res = -1): 

        self.data_path = data_path
        self.track_3d = track_3d
        self.n_frames = n_frames
        self.max_res = max_res

        # generate metadata for the provided data path (requires a specific format)
        self.metadata = self._generate_metadata(track_3d)
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

        pose = np.load(row['pose_path'])['pose']

        # if pose.shape[2] > 60:
        #     ix_p = np.random.choice(pose.shape[2], size=128)
        #     # ix_p = np.arange(60)
        #     pose = pose[:, :, ix_p, :]
        
        coords = pose[:, start_ix:end_ix, :, :]
        coords = torch.tensor(coords, dtype = torch.float32, device='cpu')
        
        res_dict = json.loads(row['res_dict'])
        new_res_dict = json.loads(row['new_res_dict'])
        scale_dict = json.loads(row['scale_dict'])

        img_path = row['img_path']
        cam_names = sorted(get_dirs(img_path))
        img_fnames = sorted(os.listdir(os.path.join(img_path, cam_names[0])))[start_ix:end_ix]
        views = []

        # if len(cam_names) > 5:
        #     ix_cams = np.random.choice(len(cam_names), size=5, replace=False)
        #     # ix_cams = np.arange(6)
        #     cam_names = [cam_names[i] for i in ix_cams]

        # create camera group from camera parameters
        if len(cam_names) == 1: 
            cgroup = None
        else: 
            cgroup = self._load_cameras(row['camera_metadata_path'], res_dict, scale_dict) 
            cgroup = cgroup.subset_cameras_names(cam_names)

            # cgroup_f = format_camera_group(cgroup, coords.device)

        # b, s, k, r = coords.shape
        # coords_flat = rearrange(coords, 'b s k r -> (b s k) r')
        # p2d_flat = cgroup.project(coords_flat.cpu().detach().numpy())
        # p2d = rearrange(p2d_flat, 'cams (b s k) r -> cams b s k r', b=b, s=s, k=k)
        # s = np.sum(np.all((p2d > 0) & (p2d < 256), axis=-1), axis=0) 
        # good = np.all(s >= 2, axis=1)

        # coords = coords[:, :, good[0]]
            
        # if coords.shape[2] > 60:
        #     ix_p = np.random.choice(coords.shape[2], size=60, replace=False)
        #     # ix_p = np.arange(60)
        #     coords = coords[:, :, ix_p, :]
        
            
        for cam_name in cam_names:
            
            imgs = []
            
            # load images from paths and resize to desired resolution
            for img_fname in img_fnames: 

                cam_img_path = os.path.join(img_path, cam_name, img_fname)
                img = cv2.imread(cam_img_path)

                if self.max_res != 1: 
                    img = cv2.resize(img, dsize = new_res_dict[cam_name])

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)

            # for i, img in enumerate(imgs):
            #     print([np.array(img).shape for img in imgs])

            views.append(torch.tensor(np.array(imgs), dtype = torch.float32))
        
        # print(views[0].shape)
        # views = [torch.stack(v, axis = 0) for v in views]

        # print(views, coords, fnums, cgroup)
        return views, coords, fnums, cgroup


    def _get_start_ixs(self, coords):

        safe = 0
        start_ixs = []

        for i in range(coords.shape[1]): 

            if safe > 0:
                safe = safe - 1 
                continue

            coords_subset = coords[:, i:i + self.n_frames, :, :]
            enough_frames = coords_subset.shape[1] == self.n_frames
            # no_nans = np.sum(~np.isfinite(coords_subset)) == 0
            
            mask = np.isfinite(coords_subset)
            visible_coords = mask.all(axis = -1).all(axis = 1).squeeze(0)

            # if no_nans and enough_frames: 
            if np.sum(visible_coords) > 0 and enough_frames:
                start_ixs.append(i)
                safe = self.n_frames - 1

        start_ixs = np.array(start_ixs)

        return start_ixs


    def _generate_metadata(self, track_3d): 
            
        rows = []
        mode = '3d' if track_3d else '2d'

        for dataset in get_dirs(self.data_path): 
            dataset_path = os.path.join(self.data_path, dataset)

            for session in get_dirs(dataset_path): 
                session_path = os.path.join(dataset_path, session)

                for trial in get_dirs(session_path):
                    # get paths to metadata, 3d pose, and images
                    trial_path = os.path.join(session_path, trial)

                    print(trial_path)
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

                    imgs = sorted(os.listdir(os.path.join(img_path, cams[0])))

                    # get starting indices 
                    coords = np.load(pose_path)[f'pose']
                    start_ixs = self._get_start_ixs(coords)

                    # n_batches = len(imgs) // self.n_frames
                    # start_ixs = np.arange(0, len(imgs), self.n_frames)[:n_batches]
                    end_ixs = start_ixs + self.n_frames

                    # add a row to the metadata that will correspond
                    # to each batch
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

        intrinsics_dict = cam_metadata['intrinsic_matrices']
        extrinsics_dict = cam_metadata['extrinsic_matrices']
        distortions_dict = cam_metadata['distortion_matrices']

        # TODO: fix preprocessing so camera names can correspond normally
        cam_names = list(intrinsics_dict.keys())
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

        cgroup = CameraGroup(cams)

        return cgroup
