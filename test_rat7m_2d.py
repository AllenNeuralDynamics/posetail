import argparse
import os
import cv2 

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

    parser.add_argument('--config-path', default = './configs/config.toml')
    parser.add_argument('--model-path')
    parser.add_argument('--video-path')
    parser.add_argument('--data-path')
    parser.add_argument('--outdir', default = './')
    
    args = parser.parse_args()

    return args


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


def main(args): 

    config = load_config(args.config_path)
    set_seeds(config.training.seed)

    model = load_checkpoint(args.config_path, args.model_path)
    model.eval()

    dataset = Rat7mDataset(
        video_path = args.video_path, 
        data_path = args.data_path, 
        n_frames = config.dataset.test.n_frames)
        
    dataloader = DataLoader(
        dataset, 
        batch_size = config.dataset.batch_size, 
        collate_fn = custom_collate_2d)

    outpath = safe_make('results')

    results_path = get_video_predictions(video_path, 
        model, dataloader, outdir, debug_ix = -1)
        
    print(f'predictions saved to {results_path}')


if __name__ == '__main__':

    args = parse_args()
    main(args)