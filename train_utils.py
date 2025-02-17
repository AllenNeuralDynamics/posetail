import glob
import os
import json 
import re
import time
import toml
import torch
import torchvision
import yaml

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, IterableDataset, DataLoader

from easydict import EasyDict
from einops import rearrange
from pytorch_memlab import MemReporter, LineProfiler, profile
from tqdm import trange, tqdm

from posetail.datasets.datasets import Rat7mDataset
from posetail.posetail.losses import get_vis_true, unroll_batch


def set_seeds(seed = 3, set_backends = True):

    np.random.seed(seed)
    torch.manual_seed(seed)

    # seeds for (multi) gpu operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # seeds for nondeterministic operations - note that this
    # could make the code less efficient
    if set_backends:
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False


def get_dataset(dataset_name, **kwargs): 

    if dataset_name == 'rat7m':
        dataset = Rat7mDataset(**kwargs)
    else:
        raise ValueError(f'no functionality for dataset named {dataset}')

    return dataset

def load_config(config_path): 
    ''' 
    loads and returns the toml configuration file in which
    keys can be accessed.like.this
    '''
    with open(config_path, 'r') as toml_file:
        config = toml.load(toml_file)

    config = EasyDict(config)

    return config

# def load_config(config_path): 
#     ''' 
#     loads and returns the toml configuration file in which
#     keys can be accessed.like.this
#     '''
#     config = {}
#     ext = os.path.splitext(config_path)[1]

#     if ext == '.yaml':
#         with open(config_path, 'r') as yaml_file:
#             config = yaml.safe_load(yaml_file)

#     elif ext == '.toml': 
#         with open(config_path, 'r') as toml_file:
#             config = toml.load(toml_file)

#     if '_wandb' in config:
#         config.pop('_wandb')

#     config = EasyDict(config)

#     return config


def save_config(exp_dir, config_name = 'config.toml'):

    config_path = os.path.join(exp_dir, config_name)

    with open(config_path, 'w') as toml_file:
        toml.dump(config, toml_file)


def write_json(json_path, results): 
    '''
    appends results to a json file
    '''
    with open(json_path, 'a') as json_file: 
        json_file.write(json.dumps(results) + '\n')


def save_checkpoint(model, optimizer, criterion, prefix, epoch): 

    checkpoint_path = os.path.join(prefix, 'checkpoints', 
        f'{checkpoint}_{str(i).zfill(6)}.pth')
    
    state_dict = {
        'epoch': i,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'criterion_state': criterion.state_dict(),
    }

    torch.save(state_dict, checkpoint_path)


def print_memory(device): 

    if torch.cuda.is_available():
        
        memory_alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
        memory_res = torch.cuda.memory_reserved(device) / 1024 ** 3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

        print(f'allocated memory: {memory_alloc:.3f} GB')
        print(f'reserved memory: {memory_res:.3f} GB')
        print(f'total memory: {memory_total:.3f} GB\n')
    
    return memory_alloc, memory_res, memory_total


def get_steps_per_epoch(dataset, dataloader):

    steps_per_epoch = 0 

    if isinstance(dataset, IterableDataset):
        for i in dataloader: 
            steps_per_epoch += 1
    else: 
        steps_per_epoch = len(dataset)

    return steps_per_epoch

# @profile
def train_epoch(model, dataloader, optimizer, loss, 
                use_amp = False, amp_type = torch.float16, 
                prefix = 'train/', debug_ix = -1): 

    device = model.device
    model.train()

    grad_scaler = GradScaler(enabled = use_amp)

    for j, batch in enumerate(dataloader):

        if j == debug_ix: 
            break
    
        views = [view.to(device) for view in batch.views]
        coords = batch.coords.to(device)
        cgroup = None 
        
        if 'cgroup' in batch: 
            cgroup = batch.cgroup

        vis = get_vis_true(coords)

        coords_true, vis_true = unroll_batch(
            coords = coords, 
            vis = vis, 
            stride = model.S, 
            stride_overlap = model.stride_overlap
        )
                                   
        # get model predictions
        if use_amp:

            with torch.autocast(device_type = device.type, dtype = amp_type, enabled = use_amp):

                outputs = model(
                    views = views, 
                    coords = coords[:, 0, ...], 
                    camera_group = cgroup, 
                    offset_dict = None
                )
                total_loss = loss(outputs, coords_true, vis_true, device = outputs[0].device)

            grad_scaler.scale(total_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad(set_to_none = True)

        else: 
            outputs = model(
                views = views, 
                coords = coords[:, 0, ...], 
                camera_group = cgroup, 
                offset_dict = None
            )
                
            total_loss = loss(outputs, coords_true, vis_true, device = outputs[0].device)
            # report = reporter.report()

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    loss_summary = loss.collapse_history(prefix = prefix)

    return loss_summary


def eval_epoch(model, dataloader,
               prefix = 'test/', debug_ix = -1): 

    device = model.device
    model.eval()

    for j, batch in enumerate(dataloader):

        if j == debug_ix: 
            break
    
        views = [view.to(device) for view in batch.views]
        coords = batch['coords'].to(device)
        cgroup = batch.cgroup

        vis = get_vis_true(coords)

        coords_true, vis_true = unroll_batch(
            coords = coords, 
            vis = vis, 
            stride = model.S, 
            stride_overlap = model.stride_overlap
        )
                                   
        # get model predictions
        coords_pred, vis_pred, *_ = model(
                views = views, 
                coords = coords[:, 0, ...], 
                camera_group = cgroup, 
                offset_dict = None
            )

        # calculate evaluation metrics
        occlusion_acc = get_occlusion_accuracy(vis_pred, vis_true)

        mpjpe = get_mpjpe(coords_pred, coords_true, vis_pred, vis_true)

        thresholds = [1, 2, 4, 8, 16]
        delta_x_avg, delta_x_dict = get_delta_x_avg(
            coords_pred, coords_true, 
            vis_pred, vis_true, 
            thresholds = thresholds
        )

        eval_matrics = {
            f'{prefix}occlusion_accuracy': occlusion_acc,
            f'{prefix}mjpje': mpjpe,
            f'{prefix}delta_x_avg': delta_x_avg,
        }

        eval_metrics.update(delta_x_dict)

    return eval_metrics