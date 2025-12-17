import os
import cv2 
import glob

import numpy as np

import torch
from train_utils import *



def get_checkpoint(wandb_prefix, run_id, checkpoint = None):

    if checkpoint is not None: 
        checkpoint_fmt = str(checkpoint).zfill(6)
        checkpoint_path = os.path.join(
            wandb_prefix, run_id, 'files', 'checkpoints', 
            f'checkpoint_{checkpoint_fmt}.pth')
        
    else:
        checkpoints = sorted(glob.glob(
            os.path.join(wandb_prefix, run_id, 'files', 'checkpoints', '*.pth')))
        checkpoint_path = checkpoints[-1]

    return checkpoint_path


def load_predictions(data_path, device):

    data = np.load(data_path) 

    coords_pred = torch.from_numpy(data['coords_pred']).to(device)
    vis_pred = torch.from_numpy(data['vis_pred']).to(device)
    conf_pred = torch.from_numpy(data['conf_pred']).to(device)

    coords_true = torch.from_numpy(data['coords_true']).to(device)
    vis_true = torch.from_numpy(data['vis_true']).to(device)

    fnums = torch.from_numpy(data['fnums']).to(device)
    video_path = ''.join(data['video_path'])

    return coords_pred, vis_pred, conf_pred, coords_true, vis_true, fnums, video_path

def format_camera(cam, device):
    return {
        'ext': torch.as_tensor(cam.get_extrinsics_mat(), dtype = torch.float32, device = device),
        'mat': torch.as_tensor(cam.get_camera_matrix(), dtype = torch.float32, device = device),
        'dist': torch.as_tensor(cam.dist, device = device, dtype = torch.float32)
    }

def format_camera_group(camera_group, device):
    return [format_camera(cam, device)
            for cam in camera_group.cameras]


def move_dict_to_device(d, device):

    new_dict = {}

    for k, v in d.items():
        if isinstance(v, torch.Tensor): 
            new_dict[k] = v.to(device)
        else: 
            new_dict[k] = v

    return new_dict

def get_video_predictions(video_paths, model, dataloader, pred_path, device, debug_ix = -1):

    torch.set_float32_matmul_precision('high')
    model.eval()

    start_time = time.time()
    timestamp = get_timestamp()

    coords_pred = []
    vis_pred = []
    conf_pred = []
    coords_true = []
    vis_true = []
    fnums = []

    for j, batch in enumerate(dataloader):

        views = [view.to(device) for view in batch.views]
        coords = batch.coords.to(device)
        fnum = batch.fnums.to(device)

        if j == debug_ix: 
            break

        cgroup = None 
        if 'cgroup' in batch: 
            cgroup = batch.cgroup
            cgroup = [move_dict_to_device(cam_dict, device) for cam_dict in cgroup]

        vis = get_vis_true(coords)
                      
        # get model predictions
        with torch.no_grad():
            coords_p, vis_p, conf_p, *_ = model(
                views = views, 
                coords = coords[:, 0, ...], 
                camera_group = cgroup, 
                offset_dict = None
            )

        coords_pred.append(torch.squeeze(coords_p, dim = 0))
        vis_pred.append(torch.squeeze(vis_p, dim = 0))
        conf_pred.append(torch.squeeze(conf_p, dim = 0))
        coords_true.append(torch.squeeze(coords, dim = 0))
        vis_true.append(torch.squeeze(vis, dim = 0))
        fnums.append(torch.squeeze(fnum, dim = 0))

    coords_pred = torch.cat(coords_pred, dim = 0)
    vis_pred = torch.cat(vis_pred, dim = 0)
    conf_pred = torch.cat(conf_pred, dim = 0)
    coords_true = torch.cat(coords_true, dim = 0)
    vis_true = torch.cat(vis_true, dim = 0)
    fnums = torch.cat(fnums, dim = 0)

    elapsed_time = time.time() - start_time
    elapsed_time_hms = str(timedelta(seconds = elapsed_time)).split('.')[0]

    np.savez(pred_path,
        coords_pred = coords_pred.cpu(), 
        vis_pred = vis_pred.cpu(), 
        conf_pred = conf_pred.cpu(),
        coords_true = coords_true.cpu(),
        vis_true = vis_true.cpu(),
        fnums = fnums.cpu(), 
        video_path = video_paths, 
        elapsed_time = list(np.array([elapsed_time])), 
        elapsed_time_hms = list(elapsed_time_hms))

    return pred_path


def pad_array(coords, n_kpts = 20): 

    coords_new = []

    for x in coords:
        n, _ = x.shape
        padding = [(0, n_kpts - n), (0, 0)]
        x_new = np.pad(x, padding, mode = 'constant', constant_values = np.nan)
        coords_new.append(x_new)

    coords_new = np.array(coords_new)

    return coords_new


def get_predictions(video_paths, model, dataloader, pred_path, device, debug_ix = -1):

    torch.set_float32_matmul_precision('high')
    model.eval()

    start_time = time.time()
    timestamp = get_timestamp()

    coords_pred = []
    vis_pred = []
    conf_pred = []
    coords_true = []
    vis_true = []
    fnums = []

    for j, batch in enumerate(dataloader):

        views = [view.to(device) for view in batch.views]
        coords = batch.coords.to(device)
        fnum = batch.fnums.to(device)

        if j == debug_ix: 
            break

        cgroup = None 
        if 'cgroup' in batch: 
            cgroup = batch.cgroup
            cgroup = [move_dict_to_device(cam_dict, device) for cam_dict in cgroup]

        vis = get_vis_true(coords)
                      
        # get model predictions
        with torch.no_grad():
            coords_p, vis_p, conf_p, *_ = model(
                views = views, 
                coords = coords[:, 0, ...], 
                camera_group = cgroup, 
                offset_dict = None
            )

        coords_pred.append(pad_array(torch.squeeze(coords_p, dim = 0).cpu().numpy()))
        # vis_pred.append(torch.squeeze(vis_p, dim = 0).cpu().numpy())
        # conf_pred.append(torch.squeeze(conf_p, dim = 0).cpu().numpy())
        coords_true.append(pad_array(torch.squeeze(coords, dim = 0).cpu().numpy()))
        # vis_true.append(torch.squeeze(vis, dim = 0).cpu().numpy())
        fnums.append(torch.squeeze(fnum, dim = 0).cpu().numpy())

    coords_pred = np.concatenate(coords_pred, axis = 0)
    # vis_pred = torch.cat(vis_pred, axis = 0)
    # conf_pred = torch.cat(conf_pred, axis = 0)
    coords_true = np.concatenate(coords_true, axis = 0)
    # vis_true = np.concatenate(vis_true, axis = 0)
    fnums = np.concatenate(fnums, axis = 0)

    elapsed_time = time.time() - start_time
    elapsed_time_hms = str(timedelta(seconds = elapsed_time)).split('.')[0]

    np.savez(pred_path,
        coords_pred = coords_pred, 
        # vis_pred = vis_pred.cpu(), 
        # conf_pred = conf_pred.cpu(),
        coords_true = coords_true,
        # vis_true = vis_true.cpu(),
        fnums = fnums, 
        video_path = video_paths, 
        elapsed_time = list(np.array([elapsed_time])), 
        elapsed_time_hms = list(elapsed_time_hms))

    return pred_path


def generate_video_2d(video_path, results_path, outpath, run_id, scale, device): 
    
    # TODO: get camera group and project coords
    (coords_pred, vis_pred, conf_pred, coords_true, 
    vis_true, fnums, video_path) = load_predictions(results_path, device)

    coords_true = coords_true.cpu().numpy().astype(int)
    coords_true[..., 0] = coords_true[..., 0] * scale[0]
    coords_true[..., 1] = coords_true[..., 1] * scale[1]

    coords_pred = coords_pred.cpu().numpy().astype(int)
    coords_pred[..., 0] = coords_pred[..., 0] * scale[0]
    coords_pred[..., 1] = coords_pred[..., 1] * scale[1]

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0 # cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    runid = run_id.split('-')[-1]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_outpath = os.path.join(outpath, f'{video_name}_{runid}.mp4')
    out = cv2.VideoWriter(video_outpath, fourcc, fps, (frame_width, frame_height))

    i = 0
    j = 0
    ret = True

    while ret:

        ret, frame = cap.read() 

        if not ret: 
            break
        
        if i not in fnums:
            out.write(frame)

        else:
            for coord_true, coord_pred in zip(coords_true[j], coords_pred[j]):
                cv2.circle(frame, tuple(coord_true), 5, (0, 255, 0), -1)
                cv2.circle(frame, tuple(coord_pred), 5, (0, 0, 255), -1)

            out.write(frame)
            j += 1

        i += 1

    cap.release()
    out.release()

    return video_outpath
