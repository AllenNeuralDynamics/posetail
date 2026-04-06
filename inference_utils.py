import os
import cv2 
import glob

import torch

import numpy as np

from collections import defaultdict

from posetail.datasets.utils import get_dirs
from train_utils import *



def get_checkpoint(wandb_prefix, run_id, checkpoint = None):

    if checkpoint is not None: 
        checkpoint_fmt = str(checkpoint).zfill(8)
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


def combine_predictions(prefix): 

    # traverse results for a particular dataset
    for session in get_dirs(prefix): 

        session_path = os.path.join(prefix, session)

        for trial in get_dirs(session_path): 

            trial_path = os.path.join(session_path, trial)

            # skip if there are no npz prediction files
            prediction_paths = sorted(glob.glob(os.path.join(trial_path, 'predictions', 'predictions_*.npz')))
            if len(prediction_paths) == 0:
                print(f'skipping... no prediction paths found at {trial_path}')
                continue

            data = [np.load(p) for p in prediction_paths] 

            # extract metadata 
            keys_to_exclude = ['coords_pred', 'vis_pred', 'conf_pred',
                               'coords_true', 'vis_true', 'fnums']

            sample_info = {k: data[0][k] for k in data[0].keys() if k not in keys_to_exclude}

            # combine predictions from each consecutive time period
            coords_pred = np.concatenate([d['coords_pred'] for d in data], axis = 0)

            vis_pred = np.concatenate([d['vis_pred'] for d in data], axis = 0)
            conf_pred = np.concatenate([d['conf_pred']for d in data], axis = 0)
            coords_true = np.concatenate([d['coords_true'] for d in data], axis = 0)
            vis_true = np.concatenate([d['vis_true'] for d in data], axis = 0)
            fnums = np.concatenate([d['fnums'] for d in data], axis = 0)

            results = {
                'coords_pred': coords_pred, 
                'vis_pred': vis_pred, 
                'conf_pred': conf_pred,
                'coords_true': coords_true,
                'vis_true': vis_true,
                'fnums': fnums, 
            }

            results.update(sample_info)

            # save combined data 
            predictions_fname = os.path.join(prefix, session, trial, f'predictions.npz')
            np.savez(predictions_fname, **results)
            print(f'predictions saved to {predictions_fname}')


def predict_on_dataset_3d(model, dataloader, outpath, device, 
                          max_kpts = 1000, debug_ix = None):

    torch.set_float32_matmul_precision('high')
    model.eval()

    for j, batch in enumerate(dataloader):

        if debug_ix and j == debug_ix: 
            break

        views = [view.to(device) for view in batch.views]
        coords = batch.coords.to(device)
        vis = batch.vis
        fnums = batch.fnums.cpu().numpy()
        cgroup = batch.cgroup 
        sample_info = batch.sample_info
        
        # fallback if visibilities are not provided
        if vis is None: 
            vis = get_vis_true(coords)

        if cgroup: 
            cgroup = [dict_to_device(cam_dict, device) for cam_dict in cgroup]
        
        # can do multiple passes if there are a lot of keypoints to predict 
        # (helps reduce memory)
        n_passes = np.ceil(coords.shape[2] / max_kpts).astype(int)
        coords_pred = []
        vis_pred = []
        conf_pred = []
        coords_true = []
        vis_true = []

        for i in range(n_passes): 

            coords_subset = coords[:, :, i * max_kpts : i * max_kpts + max_kpts, :]
            vis_subset = vis[:, :, i * max_kpts : i * max_kpts + max_kpts, :]

            # TODO: handle NaNs, don't want to pass in coords that are NaN

            # get model predictions given coords in the first frame
            with torch.no_grad():
                outputs = model(
                    views = views, 
                    coords = coords_subset[:, 0, :, :], 
                    camera_group = cgroup
                )

                coords_pred.append(torch.squeeze(outputs['coords_pred'], dim = 0).cpu().numpy())
                vis_pred.append(torch.squeeze(outputs['vis_pred'], dim = 0).cpu().numpy())
                conf_pred.append(torch.squeeze(outputs['conf_pred'], dim = 0).cpu().numpy())
                coords_true.append(torch.squeeze(coords_subset, dim = 0).cpu().numpy())
                vis_true.append(torch.squeeze(vis_subset, dim = 0).cpu().numpy())

        results = {
            'coords_pred': np.concatenate(coords_pred, axis = 1), 
            'vis_pred': np.concatenate(vis_pred, axis = 1), 
            'conf_pred': np.concatenate(conf_pred, axis = 1),
            'coords_true': np.concatenate(coords_true, axis = 1),
            'vis_true': np.concatenate(vis_true, axis = 1),
        }
        results.update({'fnums': fnums})

        keys_to_exclude = ['fnums']
        if sample_info.subject_ids is not None: 
            results.update({'subject_ids': sample_info.subject_ids})
        else: 
            keys_to_exclude.append('subject_ids')

        results.update({k: sample_info[k] for k in sample_info.keys() if k not in keys_to_exclude})

        # save predictions
        start_ix = str(results['start_ix']).zfill(8)
        predictions_outpath = os.path.join(
            outpath, results['session'], results['trial'], 'predictions')
        os.makedirs(predictions_outpath, exist_ok = True)
        predictions_fname = os.path.join(predictions_outpath, f'predictions_{start_ix}.npz')
        np.savez(predictions_fname, **results)
        print(f'predictions saved to {predictions_fname}')

    return outpath


def generate_video_2d(video_path, results_path, outpath, run_id, scale, device): 
    # NOTE: deprecated for now
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
