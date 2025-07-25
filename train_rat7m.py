#!/usr/bin/env python3

import argparse
import os
import wandb

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from pytorch_memlab import MemReporter, LineProfiler, profile

from posetail.datasets.datasets import custom_collate_2d, custom_collate_3d
from posetail.posetail.losses import *
from posetail.posetail.tracker import Tracker
from train_utils import *


''' 
python train_rat7m.py --config-path configs/config_default_2d.toml
python train_rat7m.py --config-path configs/config_default_3d.toml

'''

def parse_args(): 
    '''
    parse command line arguments
    ''' 
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', 
        default = './configs/config.toml', 
        help = 'path to model configuration file (.toml)')
    
    args = parser.parse_args()

    return args


def main(config_path): 

    config = load_config(config_path)

    set_seeds(config.training.seed)

    train_dataset = get_dataset(**config.dataset.train)

    train_loader = DataLoader(
        train_dataset, 
        batch_size = config.dataset.batch_size, 
        collate_fn = custom_collate_3d if config.model.track_3d else custom_collate_2d)

    if 'steps_per_epoch' in config.training.scheduler and config.training.scheduler.steps_per_epoch == -1:
        steps_per_epoch = get_steps_per_epoch(train_dataset, train_loader)
        config.training.scheduler.steps_per_epoch = steps_per_epoch

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

    device = torch.device(config.devices.device)
    model = Tracker(device = device, **config.model) 
    model.to(device)

    # profiler = LineProfiler(
    #     train_epoch, model, model.forward, 
    #     model.forward_iteration, model.cnn.forward, 
    #     model.corr_mlp.forward, model.tsformer.forward
    # )
    # profiler.enable()

    # reporter = MemReporter(model)
    # print(reporter.report())
    # print('')

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = config.training.optimizer.learning_rate, 
        weight_decay = config.training.optimizer.weight_decay)

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
 
    for i in range(config.training.n_epochs):

        result_dict = {'epoch': i}

        evaluate = i % config.training.eval_freq == 0

        train_dict = train_epoch(
            model = model,
            dataloader = train_loader, 
            optimizer = optimizer,
            loss = train_loss,
            scheduler = scheduler, 
            max_grad_norm = config.training.max_grad_norm,
            debug_ix = config.training.debug_ix, 
            use_amp = config.training.use_half_precision, 
            evaluate = evaluate)

        result_dict.update(train_dict)

        # if i % config.training.eval_freq == 0: 

        #     eval_dict = eval_epoch(
        #         model = model, 
        #         dataloader = val_loader,
        #         prefix = 'val/'
        #         debug_ix = self.training.debug_ix)

        #     result_dict.update(eval_dict)

        checkpoint_cond = ((i % config.training.checkpoint_freq == 0) or
                           (i + 1 == config.training.n_epochs))
        if checkpoint_cond: 
            save_checkpoint(model, optimizer, prefix = exp_dir, epoch = i)

        # log to wandb 
        wandb.log(result_dict)
        # profiler.print_stats()
        # print_memory(device)

        # save losses and evaluation metrics to json
        write_json(json_path, result_dict)
        wandb.save(json_path, base_path = exp_dir)

        if i % config.training.print_freq == 0:
            print(result_dict)
            
        train_loss.reset_history()
        val_loss.reset_history()

    wandb.finish()


if __name__ == '__main__':

    args = parse_args()
    main(args.config_path)
