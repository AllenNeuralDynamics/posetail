import argparse
import os
import cv2 
import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

from easydict import EasyDict

import torch
from torch.utils.data import DataLoader

from posetail.datasets.datasets import Rat7mDataset, Rat7mIterableDataset, custom_collate_2d
from train_utils import *


def parse_args():
    '''
    parse command line arguments
    ''' 
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-ids', nargs = '+', default = [])
    parser.add_argument('--video-path')
    parser.add_argument('--data-path')

    args = parser.parse_args()

    return args


def get_checkpoint(run_id, checkpoint = None):

    if checkpoint is not None: 
        checkpoint_fmt = str(checkpoint).zfill(6)
        checkpoint_path = f'/allen/aind/scratch/katie.rupp/wandb/{run_id}/files/checkpoints/checkpoint_{checkpoint_fmt}.pth'
    else:
        checkpoints = glob.glob(f'/allen/aind/scratch/katie.rupp/wandb/{run_id}/files/checkpoints/*.pth')
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


def get_video_predictions(video_path, model, dataloader, outdir, debug_ix = -1):

    device = model.device
    model.eval()

    start_time = time.time()
    timestamp = get_timestamp()

    coords_pred = []
    vis_pred = []
    conf_pred = []
    coords_true = []
    vis_true = []
    fnums = []

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    for j, batch in enumerate(dataloader):

        views = [view.to(device) for view in batch.views]
        coords = batch.coords.to(device)
        fnum = batch.fnums.to(device)

        if j == debug_ix: 
            break

        cgroup = None 
        if 'cgroup' in batch: 
            cgroup = batch.cgroup

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

    results_path = os.path.join(outdir, f'{video_name}_predictions.npz')
    elapsed_time = time.time() - start_time
    elapsed_time_hms = str(timedelta(seconds = elapsed_time)).split('.')[0]

    np.savez(results_path,
        coords_pred = coords_pred.cpu(), 
        vis_pred = vis_pred.cpu(), 
        conf_pred = conf_pred.cpu(),
        coords_true = coords_true.cpu(),
        vis_true = vis_true.cpu(),
        fnums = fnums.cpu(), 
        video_path = list(video_path), 
        elapsed_time = list(np.array([elapsed_time])), 
        elapsed_time_hms = list(elapsed_time_hms))

    return results_path


def generate_video(video_path, results_path, outpath, run_id, scale): 

    device = (torch.device('cuda') if torch.cuda.is_available() else 'cpu')
    (coords_pred, vis_pred, conf_pred, coords_true, 
    vis_true, fnums, video_path) = load_predictions(results_path, device)

    coords_true = coords_true.cpu().numpy().astype(int)
    coords_true[..., 0] = coords_true[..., 0] * scale[0]
    coords_true[..., 1] = coords_true[..., 1] * scale[1]

    # coords_pred = coords_pred * 2
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

    ret = True
    i = 0
    j = 0

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


def main(run_ids, video_path, data_path): 

    outpath = safe_make('results')
    figpath = safe_make('figures')

    for run_id in run_ids:

        config_path = f'/allen/aind/scratch/katie.rupp/wandb/{run_id}/files/config.toml'
        config = load_config(config_path)

        model_path = get_checkpoint(run_id, checkpoint = None)
        model = load_checkpoint(config_path, model_path)
        model.eval()

        set_seeds(config.training.seed)

        dataset = Rat7mDataset(
            video_path = video_path, 
            data_path = data_path, 
            n_frames = config.dataset.test.n_frames, 
            max_res = config.dataset.train.max_res) # TODO: add to config and change to test

        dataloader = DataLoader(
            dataset, 
            batch_size = config.dataset.batch_size, 
            collate_fn = custom_collate_2d)

        results_path = get_video_predictions(video_path, 
            model, dataloader, outpath, debug_ix = -1)
            
        print(f'predictions saved to {results_path}')

        video_outpath = generate_video(
            video_path = video_path, 
            results_path = results_path, 
            outpath = figpath, 
            run_id = run_id, 
            scale = dataloader.dataset.scale)

        print(f'video saved to {video_outpath}\n')



if __name__ == '__main__':

    # args = parse_args()

    # run_ids = args.run_ids
    # video_path = args.video_path
    # data_path = args.data_path
    
    run_ids = ['run-20250512_103400-x4ehk6ng', 'run-20250512_103343-g2gvwprc', 
               'run-20250512_103342-ykwoxm66', 'run-20250512_103342-oji10kgu']

    video_path = '/allen/aind/scratch/katie.rupp/data/rat7m/videos/s5-d2/s5-d2-camera1-0.mp4'
    data_path = '/allen/aind/scratch/katie.rupp/data/rat7m/data/mocap-s5-d2.mat'

    main(run_ids, video_path, data_path)
