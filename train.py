#!/usr/bin/env python3

import argparse
import os
import wandb

import torch
import torch.multiprocessing as mp
import torch.optim as optim
# from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler

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
pixi run python train.py --config-path configs/config_minicubes_finetuning.toml --precision 32 --devices 1 
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

    # parser.add_argument('--devices', 
    #     nargs = '*',
    #     default = 'auto', 
    #     help = 'number of gpus to use, list of gpu indices to use, or auto to use all available gpus')
    
    parser.add_argument('--devices', 
        default = 1, 
        help = 'number of gpus to train the model on')

    parser.add_argument('--strategy', 
        default = 'ddp', 
        help = 'training strategy, e.g. dp, ddp, ddp_spawn, ddp_find_unused_parameters_true, xla, deepspeed, fsdp')

    parser.add_argument('--num-nodes', 
        default = 1, 
        help = 'number of nodes to train the model on')

    parser.add_argument('--precision', 
        default = '32-true', 
        help = 'precision type with the option to use mixed precision, e.g. 32, 32-true, 16-mixed, bf16-mixed')

    args = parser.parse_args()

    return args

def run(config_path, fabric):

    mp.set_start_method('spawn', force = True)
    torch.set_float32_matmul_precision('high')

    config = load_config(config_path)
    set_seeds(config.training.seed)

    # set up training dataloader    
    train_dataset = PosetailDataset(config, split = 'train')

    sampler = DistributedSampler(
        train_dataset,
        num_replicas = fabric.world_size, 
        rank = fabric.global_rank,
        shuffle = True, 
        seed = config.training.get('seed', None)
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size = config.dataset.batch_size, 
        collate_fn = custom_collate,
        sampler = sampler,
        shuffle = False,
        num_workers = 12)

    
    train_loader = fabric.setup_dataloaders(train_loader)

    # set up validation dataloader 
    val = config.dataset.val.get('split', None)

    if val: 

        val_dataset = PosetailDataset(config, split = 'val')

        val_loader = DataLoader(
            val_dataset, 
            batch_size = config.dataset.batch_size, 
            collate_fn = custom_collate,
            shuffle = True,
            num_workers = 12)
        
        val_loader = fabric.setup_dataloaders(val_loader)

    # torch.autograd.set_detect_anomaly(True)
    
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

    # optionally load a model checkpoint 
    checkpoint_path = config.training.get('checkpoint_path', None)
    if checkpoint_path:
        print(f'loading model checkpoint {checkpoint_path}...')
        param_dict = torch.load(checkpoint_path)['model_state']
        missing_keys, unexpected_keys = model.load_state_dict(param_dict, strict = False)
        print(f'received missing keys: {missing_keys}')
        print(f'received unexpected keys: {unexpected_keys}')

    # compile the model
    model.cnn.compile()
    model.corr_mlp.compile()
    model.tsformer.compile()

    if model.mode_3d == 'minicubes':
        for v2v in model.minicube_v2v:
            v2v.compile()
    elif model.mode_3d == 'triplane':
        model.triplane_cnn.compile()
        
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

    # put metrics in terms of one gpu, since all logging/checkpointing 
    # will happen on the zero rank gpu
    iters_per_gpu = total_to_per_gpu(config.training.n_iterations, fabric.world_size)
    checkpoint_freq = total_to_per_gpu(config.training.checkpoint_freq, fabric.world_size) 
    eval_metric_freq = total_to_per_gpu(config.training.eval_metric_freq, fabric.world_size)
    val_freq = total_to_per_gpu(config.training.val_freq, fabric.world_size)
    print_freq = total_to_per_gpu(config.training.print_freq, fabric.world_size) 
    
    # set up LR scheduler
    scheduler = None
    if config.training.scheduler_type == 'onecyclelr': 
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer = optimizer,
            max_lr = config.training.optimizer.learning_rate,
            total_steps = iters_per_gpu,
            **config.training.scheduler)

    elif config.training.scheduler_type == 'multisteplr': 
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer = optimizer, 
            **config.training.scheduler)

    train_loss = TotalLoss(**config.training.losses)
    val_loss = TotalLoss(**config.training.losses)
    
    # total_params = sum(p.numel() for p in model.parameters())
    # print(total_params)

    train_iter = iter(train_loader)

    for i in range(iters_per_gpu):

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        global_i = i * fabric.world_size + fabric.local_rank
        result_dict = {'iteration': global_i}
        evaluate = i % eval_metric_freq == 0

        train_dict = train_iteration(
            config = config,
            model = model,
            fabric = fabric,
            batch = batch, 
            optimizer = optimizer,
            loss = train_loss,
            scheduler = scheduler, 
            evaluate = evaluate)

        result_dict.update(train_dict)

        # evaluate model on validation dataset
        if val and i % val_freq == 0: 

            val_dict = test_epoch(
                config = config,
                model = model, 
                dataloader = val_loader,
                loss = val_loss,
                prefix = 'val/',
                evaluate = evaluate)

            result_dict.update(val_dict)

        # log losses and eval metrics to wandb and print to console 
        if fabric.is_global_zero:

            wandb.log(result_dict)
            write_json(json_path, result_dict)
            wandb.save(json_path, base_path = exp_dir)

            if i % print_freq == 0:
                print(result_dict)
                
        # save a model checkpoint when the condition is met
        checkpoint_cond = ((i % checkpoint_freq == 0) or
                           (i + 1 == iters_per_gpu))

        if checkpoint_cond and fabric.is_global_zero:
            save_checkpoint(model, optimizer, prefix = exp_dir, i = global_i)

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


