import os
import json
import time
import toml
import torch
import yaml

import numpy as np

# from torch.cuda.amp import GradScaler
# from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, IterableDataset

from datetime import datetime, timezone, timedelta
from easydict import EasyDict
# from pytorch_memlab import MemReporter, LineProfiler, profile

# from posetail.datasets.datasets import Rat7mIterableDataset
from posetail.datasets.utils import safe_make
from posetail.posetail.eval_metrics import get_eval_metrics
from posetail.posetail.losses import get_vis_true, unroll_batch
from posetail.posetail.tracker import Tracker 

from einops import rearrange

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
        dataset = Rat7mIterableDataset(**kwargs)
    else:
        raise ValueError(f'no functionality for dataset named {dataset}')

    return dataset

def load_config(config_path, easy = True): 
    ''' 
    loads and returns the toml configuration file in which
    keys can be accessed.like.this
    '''
    with open(config_path, 'r') as toml_file:
        config = toml.load(toml_file)

    if easy: 
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


def save_config(config_path, new_config_path):

    config = load_config(config_path, easy = False)

    with open(new_config_path, 'w') as toml_file:
        toml.dump(config, toml_file)
        
        
def write_json(json_path, results): 
    '''
    appends results to a json file
    '''
    with open(json_path, 'a') as json_file: 
        json_file.write(json.dumps(results) + '\n')


def save_checkpoint(model, optimizer, prefix, epoch): 

    checkpoint_dir = safe_make(os.path.join(prefix, 'checkpoints'))

    checkpoint_path = os.path.join(checkpoint_dir, 
        f'checkpoint_{str(epoch).zfill(6)}.pth')
    
    state_dict = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }

    torch.save(state_dict, checkpoint_path)


def load_checkpoint(config_path, checkpoint_path):

    config = load_config(config_path) 

    device = torch.device(config.devices.device)

    if not torch.cuda.is_available(): 
        device = torch.device('cpu')

    model = Tracker(**config.model) 
    model.to(device)

    param_dict = torch.load(checkpoint_path, map_location = device)['model_state']
    model.load_state_dict(param_dict)

    return model


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

def get_timestamp(): 

    tz = timezone(timedelta(hours = -8))
    timestamp = datetime.now(tz)
    timestamp_fmt = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    return timestamp_fmt

def format_camera(cam, device):
    return {
        "name": cam.get_name(),
        "ext": torch.as_tensor(cam.get_extrinsics_mat(), device=device, dtype=torch.float),
        "mat": torch.as_tensor(cam.get_camera_matrix(), device=device, dtype=torch.float),
        "dist": torch.as_tensor(cam.dist, device=device, dtype=torch.float)
    }

def format_camera_group(camera_group, device):
    return [format_camera(cam, device)
            for cam in camera_group.cameras]

def dict_to_device(dd, device):
    dout = dict()
    for k, v in dd.items():
        if isinstance(v, torch.Tensor):
            dout[k] = v.to(device)
        else:
            dout[k] = v
    return dout

# @profile
def train_epoch(config, model, fabric, dataloader, 
                optimizer, loss, scheduler = None,
                prefix = 'train/',  evaluate = False): 

    device = model.device
    model.train()

    start_time = time.time()
    timestamp = get_timestamp()

    learning_rate = optimizer.param_groups[0]['lr']

    n_batches = 0
    n_frames = 0
    metric_dicts = []
    
    for j, batch in enumerate(dataloader):

        if j == config.training.debug_ix: 
            break
    
        views = [view.to(device) for view in batch.views]
        coords = batch.coords.to(device)
        cgroup = None 
        
        if 'cgroup' in batch: 
            cgroup = batch.cgroup
            # cgroup = dict_to_device(cgroup, device)

        vis = get_vis_true(coords)

        coords_true, vis_true = unroll_batch(
            coords = coords, 
            vis = vis, 
            stride = model.S, 
            stride_overlap = model.stride_overlap)

        optimizer.zero_grad()

        outputs = model(
            views = list(views), 
            coords = coords[:, 0, ...], 
            camera_group = cgroup, 
            offset_dict = None)

        coords_pred = outputs[0]
        vis_pred = outputs[1]
        conf_pred = outputs[2]

        feature_planes_levels = outputs[6]

        outputs = list(outputs[:6])

        feature_loss = model.get_feature_loss(
            feature_planes_levels = feature_planes_levels, 
            coords_full = coords, 
            camera_group = cgroup, 
            offset_dict = None)
        outputs.append(feature_loss)

        b, s, n, r = coords.shape
        coords_flat = rearrange(coords, 'b s n r -> (b s n) r')
        ixs_perm = torch.randperm(coords_flat.shape[0])
        coords_shuffle = rearrange(coords_flat[ixs_perm], '(b s n) r -> b s n r', b=b, s=s, n=n)

        bad_feature_loss = model.get_feature_loss(
            feature_planes_levels = feature_planes_levels, 
            coords_full = coords_shuffle, 
            camera_group = cgroup, 
            offset_dict = None)
        outputs.append(1 - bad_feature_loss)
        
        total_loss = loss(outputs, coords_true, vis_true, device = coords_pred.device)

        # if not torch.any(torch.isnan(total_loss)):
            # report = reporter.report()

        if torch.any(torch.isnan(total_loss)):
            print(total_loss)
            
        fabric.backward(total_loss)

        bad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2)
                if not torch.isfinite(grad_norm):
                    print(f"{name}: {grad_norm}")
                    bad = True

        if bad:
            import pickle
            outname = '/data/results/lili/posetail/test/modeltest.pth'
            state_dict = {'views': views,
                          'coords': coords,
                          'camera_group': cgroup,
                          'model_state': model.state_dict(),
                          'optimizer_state': optimizer.state_dict()}
            torch.save(state_dict, outname)
            print("torch state dumped")

        fabric.clip_gradients(model, optimizer, 
            max_norm = config.training.max_grad_norm, 
            error_if_nonfinite = True)


        optimizer.step()
        optimizer.zero_grad()
        # else:
        #     print('WARNING: nan loss')

        
        if evaluate:
            metrics_dict = get_eval_metrics(
                vis_pred = outputs[1], 
                vis_true = vis, 
                coords_pred = outputs[0], 
                coords_true = coords,
                prefix = prefix
            ) 
            metric_dicts.append(metrics_dict)

        n_batches += 1
        n_frames += coords.shape[1]

    # print_memory(device)

    if scheduler: 
        scheduler.step()
        learning_rate = scheduler.get_last_lr()[0]

    loss_dict = loss.collapse_history(prefix = prefix)

    # track time of training loop
    elapsed_time = time.time() - start_time
    elapsed_time_hms = str(timedelta(seconds = elapsed_time)).split('.')[0]

    train_dict = {f'{prefix}timestamp': timestamp,
                  f'{prefix}elapsed_time': elapsed_time,
                  f'{prefix}elapsed_time_hms': elapsed_time_hms,
                  f'{prefix}batches_per_epoch': n_batches,
                  f'{prefix}frames_per_epoch': n_frames, 
                  f'{prefix}learning_rate': learning_rate}
    train_dict.update(loss_dict)

    # average evaluation metrics if we evaluated
    if evaluate: 

        avg_metrics_dict = {}
        metrics = list(metric_dicts[0].keys())

        for metric in metrics: 
            metric_list = [float(metric_dict[metric]) for metric_dict in metric_dicts]
            avg_metrics_dict[f'{metric}_avg'] = float(np.mean(metric_list))
            avg_metrics_dict[f'{metric}_std'] = float(np.std(metric_list))
            
        train_dict.update(avg_metrics_dict)

    return train_dict


def eval_epoch(model, dataloader, loss = None, prefix = 'test/', debug_ix = -1): 

    device = model.device
    model.eval()

    start_time = time.time()
    timestamp = get_timestamp()

    metric_dicts = []

    n_batches = 0
    n_frames = 0

    for j, batch in enumerate(dataloader):

        if j == debug_ix: 
            break
    
        views = [view.to(device) for view in batch.views]
        coords = batch['coords'].to(device)
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
        with torch.no_grad():
            outputs = model(
                    views = views, 
                    coords = coords[:, 0, ...], 
                    camera_group = cgroup, 
                    offset_dict = None
                )
        
        if loss is not None:
            total_loss = loss(outputs, coords_true, vis_true, device = outputs[0].device)


        metrics_dict = get_eval_metrics(
            vis_pred = outputs[1], 
            vis_true = vis, 
            coords_pred = outputs[0], 
            coords_true = coords,
            prefix = prefix
        ) 
        metric_dicts.append(metrics_dict)

        n_batches += 1
        n_frames += coords.shape[1]

    # track time of evaluation loop
    elapsed_time = time.time() - start_time
    elapsed_time_hms = str(timedelta(seconds = elapsed_time)).split('.')[0]

    # collate evaluation data
    eval_dict = {f'{prefix}timestamp': timestamp,
                 f'{prefix}elapsed_time': elapsed_time,
                 f'{prefix}elapsed_time_hms': elapsed_time_hms, 
                 f'{prefix}batches_per_epoch': n_batches,
                 f'{prefix}frames_per_epoch': n_frames}

    if loss is not None:
        loss_dict = loss.collapse_history(prefix = prefix)
        eval_dict.update(loss_dict)

    avg_metrics_dict = {}
    metrics = list(metric_dicts[0].keys())

    for metric in metrics: 
        metric_list = [float(metric_dict[metric]) for metric_dict in metric_dicts]
        avg_metrics_dict[f'{metric}_avg'] = float(np.mean(metric_list))
        avg_metrics_dict[f'{metric}_std'] = float(np.std(metric_list))

    eval_dict.update(avg_metrics_dict)

    return eval_dict
