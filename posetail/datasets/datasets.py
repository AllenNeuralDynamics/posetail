import glob
import os 

import cv2
import scipy
import numpy as np

import torch
from torch.utils.data import Dataset, IterableDataset

from aniposelib.cameras import CameraGroup, Camera
from easydict import EasyDict as edict

from posetail.datasets.utils import extract_name, extract_num, scale_coords


def custom_collate_2d(batch):
    ''' 
    custom collate functon to enable returning 
    non-tensor, non-list, etc type objects from 
    the default collate function
    '''
    batch = list(zip(*batch))
    
    views = [torch.stack(batch[0], dim = 0)]
    coords = torch.stack(batch[1], axis = 0)
    fnums = torch.stack(batch[2], axis = 0)

    batch = edict({'views': views, 
                   'coords': coords, 
                   'fnums': fnums})

    return batch


def custom_collate_3d(batch):
    ''' 
    custom collate functon to enable returning 
    non-tensor, non-list, etc type objects from 
    the default collate function
    '''
    batch = list(zip(*batch))

    views = [torch.stack(v, dim = 0) for v in zip(*list(batch[0]))]
    coords = torch.stack(batch[1], axis = 0)
    fnums = torch.stack(batch[2], axis = 0)
    cgroup = batch[3][0]

    batch = edict({'views': views, 
                   'coords': coords,
                   'fnums': fnums, 
                   'cgroup': cgroup})

    return batch
    

class MultiviewDataset(Dataset):

    def __init__(self, video_paths, coords_path, 
                 n_frames, n_kpts, transform = None): 

        self.transform = transform

        # TODO: load coordinates from coord_path
        self.n_kpts = n_kpts
        self.caps = self.load_videos(video_paths)
        # self.coords = self.generate_coords()

        self.current_frame = 0
        self.n_frames = n_frames
        self.max_frames = self.get_max_frame()

        # set a seed for now
        self.trng = torch.Generator()
        self.trng.manual_seed(3)


    def __len__(self):

        length = self.max_frames // self.n_frames

        if np.mod(self.max_frames, self.n_frames) > 0: 
            length = length + 1

        return length

    def __getitem__(self, idx): 

        views = []
        coords = self.generate_coords()

        for i in range(len(self.caps)):

            frames = []

            for j in range(self.n_frames): 
                ret, frame = self.caps[i].read()
                frames.append(frame)

            views.append(torch.tensor(np.array(frames), dtype = torch.float32))

        self.current_frame += self.n_frames
        if self.current_frame + self.n_frames >= self.max_frames:
            self.n_frames = self.max_frames - self.current_frame

        return views, coords

    def generate_coords(self): 

        params = {'low': 100, 
                  'high': 700, 
                  'size': (self.n_frames, self.n_kpts), 
                  'generator': self.trng}

        xs = torch.randint(**params) 
        ys = torch.randint(**params)
        zs = torch.randint(**params)

        coords = torch.stack((xs, ys, zs), dim = -1)

        return coords

    def load_videos(self, video_paths): 

        caps = [cv2.VideoCapture(path) for path in video_paths]

        return caps

    def get_max_frame(self): 

        n_frames = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in self.caps]
        max_frame = np.min(n_frames)

        return max_frame


def load_pose3d(fname):

    mat = scipy.io.loadmat(fname)
    d = dict(zip(mat['mocap'].dtype.names, mat['mocap'].item()))
    bodyparts = mat['mocap'].dtype.names

    arr = []
    for bp in bodyparts:
        arr.append(d[bp])

    arr = np.array(arr)
    pose3d = arr.swapaxes(0, 1) # (time, bodyparts, 3)

    return {'pose': pose3d, 'bodyparts': bodyparts}


def make_M(rvec, tvec):

    out = np.zeros((4,4))
    # rotmat, _ = cv2.Rodrigues(np.array(rvec))
    out[:3,:3] = rvec
    out[:3, 3] = np.array(tvec).flatten()
    out[3, 3] = 1

    return out

def load_calibration(calib_file):

    mat = scipy.io.loadmat(calib_file)

    intrinsics = []
    extrinsics = []
    distortions = []

    names = [x.lower() for x in mat['cameras'].dtype.names]
    cam_order = [0, 1, 4, 2, 3, 5]

    cam_objs = []

    for i, cam in enumerate(cam_order):
        
        intrin = mat['cameras'].item()[cam]['IntrinsicMatrix'].item().transpose()

        extrin = make_M(mat['cameras'].item()[cam]['rotationMatrix'].item().transpose(),
                                mat['cameras'].item()[cam]['translationVector'].item())

        rotmat = mat['cameras'].item()[cam]['rotationMatrix'].item().transpose()

        distort = [mat['cameras'].item()[cam]['RadialDistortion'].item()[0,0],
                mat['cameras'].item()[cam]['RadialDistortion'].item()[0,1],
                mat['cameras'].item()[cam]['TangentialDistortion'].item()[0,0],
                mat['cameras'].item()[cam]['TangentialDistortion'].item()[0,1], 0.0]

        intrinsics.append(intrin)
        extrinsics.append(extrin)
        distortions.append(distort)

        rvec = cv2.Rodrigues(rotmat)[0].T[0]
        tvec = mat['cameras'].item()[cam]['translationVector'].item()[0]

        cam_obj = Camera(matrix = intrin,
                        dist = distort,
                        rvec = rvec, 
                        tvec = tvec,
                        name = names[cam])

        cam_objs.append(cam_obj)

    cgroup = CameraGroup(cam_objs)

    return cgroup


class Rat7mDataset(Dataset): 

    def __init__(self, video_path, data_path, n_frames): 

        self.video_path = video_path
        self.data_path = data_path
        self.n_frames = n_frames

        self.subject, self.cam, self.start_frame = self._parse_video_name()

        self.cgroup = load_calibration(self.data_path)
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.coords2d = self._load_coords()
        self.start_ixs = self._get_start_ixs()


    def _parse_video_name(self):

        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        subject, camera_name, start_frame = video_name.rsplit('-', 2)

        start_frame = int(start_frame)
        cam_num = int(camera_name.lstrip('camera'))

        cam_order = [0, 1, 4, 2, 3, 5]
        cam_dict = dict(zip(range(1, len(cam_order) + 1), cam_order))
        cam = cam_dict[cam_num]

        return subject, cam, start_frame


    def _load_coords(self): 

        coords3d = load_pose3d(self.data_path)['pose']
        n_time, n_kpts, _ = coords3d.shape 

        coords2d = self.cgroup.project(coords3d)
        n_cameras = coords2d.shape[0]
        coords2d = coords2d.reshape(n_cameras, n_time, n_kpts, -1)

        coords2d = coords2d[self.cam, self.start_frame: self.start_frame + self.total_frames, :]

        return coords2d


    def _get_start_ixs(self):

        safe = 0
        start_ixs = []

        for i in range(len(self.coords2d)): 

            if safe > 0:
                safe = safe - 1 
                continue

            coords_subset = self.coords2d[i:i + self.n_frames, :, :]
            enough_frames = coords_subset.shape[0] == self.n_frames
            no_nans = np.sum(~np.isfinite(coords_subset)) == 0

            if no_nans and enough_frames: 
                start_ixs.append(i)
                safe = self.n_frames - 1

        return start_ixs


    def __len__(self):
        return len(self.start_ixs)


    def __getitem__(self, idx): 

        views = []
        start_frame = self.start_ixs[idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(self.n_frames):
            ret, frame = self.cap.read()
            views.append(torch.tensor(np.array(frame), dtype = torch.float32))

        views = torch.stack(views, axis = 0)

        coords = self.coords2d[start_frame:start_frame + self.n_frames, :, :]
        coords = torch.tensor(coords, dtype = torch.float32)

        fnums = torch.arange(start_frame, start_frame + self.n_frames)

        return views, coords, fnums


class Rat7mIterableDataset(IterableDataset): 

    def __init__(self, prefix, n_frames, project_2d = False, 
                sub_pattern = r's([0-5])-d([1-2])', 
                camera_pattern = r'camera([0-6])', 
                fnum_pattern = r'-(\d+).mp4', 
                max_res = None): 

        self.prefix = prefix
        self.n_frames = n_frames
        self.project_2d = project_2d

        self.sub_pattern = sub_pattern
        self.cam_pattern = camera_pattern
        self.fnum_pattern = fnum_pattern
        
        self.max_res = max_res

        self.camera_size_dict = {}


    def get_video_groups_2d(self, subject_files, sub):

        camera_names = set([extract_name(fname, self.cam_pattern) for fname in subject_files])
        video_groups = {}

        for cam in sorted(camera_names):
            
            camera_files = glob.glob(f'{self.prefix}/*/{sub}/{sub}-{cam}*.mp4', 
                recursive = True)

            camera_files_sorted = sorted(camera_files, 
                key = lambda x: extract_num(x, self.fnum_pattern))
            
            fnums = [extract_num(camera_file, self.fnum_pattern) 
                for camera_file in camera_files_sorted]

            video_groups[cam] = dict(zip(fnums, camera_files_sorted))
        
        return video_groups


    def get_video_groups_3d(self, subject_files, sub):

        camera_names = set([extract_name(fname, self.cam_pattern) for fname in subject_files])
        video_groups = []

        for cam in sorted(camera_names):
            
            camera_files = glob.glob(f'{self.prefix}/*/{sub}/{sub}-{cam}*.mp4', 
                recursive = True)

            camera_files_sorted = sorted(camera_files, 
                key = lambda x: extract_num(x, self.fnum_pattern))
                                         
            video_groups.append(camera_files_sorted)
        
        video_groups = list(zip(*video_groups))
        fnums = [extract_num(group[0], self.fnum_pattern) for group in video_groups]
        video_groups = dict(zip(fnums, video_groups))
        
        return video_groups
        
    def store_camera_size(self, cap, ix):

        if ix not in self.camera_size_dict:

            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            current_max_res = max(height, width)
            scale = self.max_res / current_max_res

            new_res = (round(width * scale), round(height * scale))

            self.camera_size_dict[ix] = {'orig_res': (width, height),
                                         'scale': scale,
                                         'new_res': new_res}


    def generate2d(self):

        data_files = glob.glob(f'{self.prefix}/**/*.mat', recursive = True)
        subjects = [extract_name(df, self.sub_pattern) for df in data_files]
        subjects = [sub for sub in subjects if sub != '']

        for sub in subjects: 

            data_path = glob.glob(f'{self.prefix}/*/mocap-{sub}.mat')[0]
            cgroup = load_calibration(data_path)

            camera_names = cgroup.get_names()
            n_cameras = len(camera_names)
            camera_dict = dict(zip(camera_names, range(n_cameras)))

            coords3d = load_pose3d(data_path)['pose']
            n_time, n_kpts, _ = coords3d.shape 

            coords2d = cgroup.project(coords3d)
            coords2d = coords2d.reshape(n_cameras, n_time, n_kpts, -1)

            subject_files = glob.glob(f'{self.prefix}/*/{sub}/{sub}*.mp4', recursive = True)
            video_groups = self.get_video_groups_2d(subject_files, sub)

            for cam, video_path_dict in video_groups.items(): 

                views = []
                camera_ix = camera_dict[cam]

                for fnum, video_path in video_path_dict.items():
                    
                    cap = cv2.VideoCapture(video_path)
                    ret = True

                    # store original camera size
                    self.store_camera_size(cap, camera_ix)

                    while ret:

                        ret, frame = cap.read()
                        
                        if not ret: 
                            break

                        if self.max_res is not None:
                            new_res = self.camera_size_dict[camera_ix]['new_res']
                            frame = cv2.resize(frame, dsize = new_res)

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        views.append(torch.tensor(np.array(frame), dtype = torch.float32))
                        fnum += 1

                        if len(views) >= self.n_frames: 

                            views = torch.stack(views, axis = 0)
                            coords = coords2d[camera_ix, fnum - self.n_frames:fnum, :, :]
                            coords = torch.tensor(coords, dtype = torch.float32)

                            # scale coordinates
                            orig_res = self.camera_size_dict[camera_ix]['orig_res']
                            scale = self.camera_size_dict[camera_ix]['scale']
                            coords = coords * scale

                            # mask coordinates
                            visible_kpts_mask = torch.isfinite(torch.sum(coords, dim = (0, 2)))
                            coords_masked = coords[:, visible_kpts_mask, :]

                            # get the corresponding fnums
                            fnums = torch.arange(fnum - self.n_frames, fnum)
                            
                            if coords_masked.shape[1] > 0: 
                                yield views, coords_masked, fnums

                            # reset for next span of frames
                            views = []

                    break # TODO: remove later, just for testing on one video
                break


    def generate3d(self):

        data_files = glob.glob(f'{self.prefix}/**/*.mat', recursive = True)
        subjects = [extract_name(df, self.sub_pattern) for df in data_files]
        subjects = [sub for sub in subjects if sub != '']

        for sub in subjects: 

            data_path = glob.glob(f'{self.prefix}/*/mocap-{sub}.mat')[0]
            cgroup = load_calibration(data_path)
            coords3d = load_pose3d(data_path)['pose']

            subject_files = glob.glob(f'{self.prefix}/*/{sub}/{sub}*.mp4', recursive = True)
            video_groups = self.get_video_groups_3d(subject_files, sub)

            views = []

            for fnum, video_paths in video_groups.items(): 
                
                caps = [cv2.VideoCapture(path) for path in video_paths]
                ret = True

                # store original camera size
                for i in range(len(caps)):
                    self.store_camera_size(caps[i], i)

                while ret:

                    frames = []

                    for i in range(len(caps)):

                        ret, frame = caps[i].read()
                        
                        if not ret: 
                            break

                        if self.max_res is not None:
                            new_res = self.camera_size_dict[i]['new_res']
                            frame = cv2.resize(frame, dsize = new_res)

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)

                    if not ret:
                        break

                    views.append(torch.tensor(np.array(frames), dtype = torch.float32))
                    fnum += 1

                    if len(views) >= self.n_frames: 

                        coords = coords3d[fnum - self.n_frames:fnum, :, :]
                        coords = torch.tensor(coords, dtype = torch.float32)
                        views = [torch.stack(list(v), axis = 0) for v in zip(*views)]

                        visible_kpts_mask = torch.isfinite(torch.sum(coords, dim = (0, 2)))
                        coords_masked = coords[:, visible_kpts_mask, :]

                        fnums = torch.arange(fnum - self.n_frames, fnum) + 1

                        for i, cam in enumerate(cgroup.cameras): 

                            orig_res = self.camera_size_dict[i]['orig_res']
                            scale = self.camera_size_dict[i]['scale']

                            cam.set_size(orig_res)
                            cam.resize_camera(scale)

                        if coords_masked.shape[1] > 0: 
                            yield views, coords_masked, fnums, cgroup
        
                        views = []


    def generate(self): 
        ''' 
        if project_2d is true, returns a single view 
        and the corresponding 2d coordinates 
        
        if project_2d is false, returns all camera views 
        according to the regex, the corresponding 3d coordinates,
        and the camera group
        ''' 

        if self.project_2d: 
            return self.generate2d()
        else: 
            return self.generate3d()


    def __iter__(self):
        sample = iter(self.generate())
        return sample