#!/usr/bin/env python3

import argparse
import os
import wandb

import torch
import torch.optim as optim
# from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from lightning.fabric import Fabric

# from pytorch_memlab import MemReporter, LineProfiler, profile

from posetail.datasets.posetail_dataset import PosetailDataset, custom_collate
from posetail.posetail.losses import *
from posetail.posetail.tracker import Tracker
from train_utils import *


''' 
python train.py --config-path configs/config_default_2d.toml
python train.py --config-path configs/config_default_3d.toml --devices 1
python train.py --config-path configs/config_default_3d.toml --devices 1 2
pixi run python train.py --config-path configs/config_default_3d.toml --precision 32 --devices 1 
'''

def parse_args(): 
    '''
    parse command line arguments
    ''' 
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', 
        default = './configs/config.toml', 
        help = 'path to model configuration file (.toml)')

    parser.add_argument('--accelerator', 
        default = 'gpu', 
        help = 'accelerator to use for training: cpu, gpu, tpu, auto')

    parser.add_argument('--devices', 
        type = int, 
        nargs = '+',
        default = [0], 
        help = 'list of gpu numbers to use')
    
    # parser.add_argument('--devices', 
    #     default = 1, 
    #     help = 'number of gpus to train the model on')

    parser.add_argument('--strategy', 
        default = 'ddp', 
        help = 'training strategy, e.g. dp, ddp, ddp_spawn, ddp_find_unused_parameters_true, xla, deepspeed, fsdp')

    parser.add_argument('--num_nodes', 
        default = 1, 
        help = 'number of nodes to train the model on')

    parser.add_argument('--precision', 
        default = '32-true', 
        help = 'precision type with the option to use mixed precision, e.g. 32, 32-true, 16-mixed, bf16-mixed')

    args = parser.parse_args()

    return args

def format_sample_input(x):

    if isinstance(x, int): 
        return x
    elif isinstance(x, list): 
        return tuple(x) 
    else: 
        return None

def run(config_path, fabric):

    torch.set_float32_matmul_precision('high')

    config = load_config(config_path)
    set_seeds(config.training.seed)

    # set up training dataloader
    cams_to_sample = format_sample_input(config.dataset.train.get('cams_to_sample', None))
    kpts_to_sample = format_sample_input(config.dataset.train.get('kpts_to_sample', None))

    train_dataset = PosetailDataset(
        data_path = config.dataset.prefix, 
        split = config.dataset.train.split,
        track_3d = config.model.track_3d, 
        n_frames = config.dataset.train.n_frames,
        max_res = config.dataset.train.max_res, 
        cams_to_sample = cams_to_sample, 
        kpts_to_sample = kpts_to_sample)

    train_loader = DataLoader(
        train_dataset, 
        batch_size = config.dataset.batch_size, 
        collate_fn = custom_collate,
        shuffle = True,
        num_workers = 8)
    
    train_loader = fabric.setup_dataloaders(train_loader)

    # set up validation dataloader 
    val = config.dataset.val.get('split', None)

    if val: 

        cams_to_sample = format_sample_input(config.dataset.val.get('cams_to_sample', None))
        kpts_to_sample = format_sample_input(config.dataset.val.get('kpts_to_sample', None))

        val_dataset = PosetailDataset(
            data_path = config.dataset.prefix,
            split = config.dataset.val.split, 
            track_3d = config.model.track_3d, 
            n_frames = config.dataset.val.n_frames,
            max_res = config.dataset.val.max_res, 
            cams_to_sample = cams_to_sample, 
            kpts_to_sample = kpts_to_sample)

        val_loader = DataLoader(
            val_dataset, 
            batch_size = config.dataset.batch_size, 
            collate_fn = custom_collate,
            shuffle = True,
            num_workers = 8)
        
        val_loader = fabric.setup_dataloaders(val_loader)

    torch.autograd.set_detect_anomaly(True)
    
    if fabric.is_global_zero:
        wandb.init(
            project = config.wandb.project_name,  
            dir = config.wandb.path, 
            mode = config.wandb.mode, 
            config = config)

        exp_dir = wandb.run.dir
        json_path = os.path.join(exp_dir, 'results.json')

        wandb_config_path = os.path.join(exp_dir, 'config.toml')
        save_config(config_path, wandb_config_path)
        wandb.save(wandb_config_path, base_path = exp_dir)

    # device = torch.device(config.devices.device)
    model = Tracker(**config.model)
    model.cnn.compile()
    model.corr_mlp.compile()
    model.tsformer.compile()

    if model.mode_3d == 'minicubes':
        model.minicube_v2v.compile()

    model = fabric.setup(model)
    model.mark_forward_method('get_feature_loss')
    
    # NOTE: memory profiling causes a CPU memory leak
    # profiler = LineProfiler(
    #     train_epoch, model, model.forward, 
    #     model.forward_iteration, model.cnn.forward, 
    #     model.corr_mlp.forward, model.tsformer.forward
    # )
    # profiler.enable()

    # reporter = MemReporter(model)
    # print(reporter.report())
    # print('')

    # set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = config.training.optimizer.learning_rate, 
        weight_decay = config.training.optimizer.weight_decay,
        amsgrad = config.training.optimizer.amsgrad,
        fused = True)

    optimizer = fabric.setup_optimizers(optimizer)

    # set up LR scheduler
    scheduler = None
    if config.training.scheduler_type == 'onecyclelr': 
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer = optimizer,
            max_lr = config.training.optimizer.learning_rate,
            epochs = config.training.n_epochs,
            **config.training.scheduler)

    elif config.training.scheduler_type == 'multisteplr': 
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer = optimizer, 
            **config.training.scheduler)

    train_loss = TotalLoss(**config.training.losses)
    val_loss = TotalLoss(**config.training.losses)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(total_params)
 
    # train model on training dataset
    for i in range(config.training.n_epochs):

        result_dict = {'epoch': i}
        evaluate = i % config.training.eval_freq == 0

        train_dict = train_epoch(
            config = config,
            model = model,
            fabric = fabric,
            dataloader = train_loader, 
            optimizer = optimizer,
            loss = train_loss,
            scheduler = scheduler, 
            evaluate = evaluate)

        result_dict.update(train_dict)

        # evaluate model on validation dataset
        if val and i % config.training.eval_freq == 0: 

            eval_dict = eval_epoch(
                config = config,
                model = model, 
                dataloader = val_loader,
                loss = val_loss,
                prefix = 'val/',
                evaluate = evaluate)

            result_dict.update(eval_dict)

        # log to wandb and print to console 
        if fabric.is_global_zero:
            wandb.log(result_dict)

            if i % config.training.print_freq == 0:
                print(result_dict)

        # save a model checkpoint when the condition is met
        checkpoint_cond = ((i % config.training.checkpoint_freq == 0) or
                           (i + 1 == config.training.n_epochs))

        if checkpoint_cond and fabric.is_global_zero:
            save_checkpoint(model, optimizer, prefix = exp_dir, epoch = i)

            # profiler.print_stats()
            # print_memory(device)

            # save losses and evaluation metrics to json
            write_json(json_path, result_dict)
            wandb.save(json_path, base_path = exp_dir)
            
        train_loss.reset_history()
        val_loss.reset_history()

    wandb.finish()


if __name__ == '__main__':

    args = parse_args()

    fabric = Fabric(
        accelerator = args.accelerator,
        devices = args.devices, 
        strategy = args.strategy,
        num_nodes = args.num_nodes,
        precision = args.precision)

    fabric.launch()
    run(config_path = args.config_path, fabric = fabric)


