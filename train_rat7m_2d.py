import argparse
import os
import wandb

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from pytorch_memlab import MemReporter, LineProfiler, profile

from posetail.datasets.datasets import custom_collate_2d
from posetail.posetail.losses import *
from posetail.posetail.tracker import Tracker
from train_utils import *


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


def main(config): 

    set_seeds(config.training.seed)

    train_dataset = get_dataset(**config.dataset.train)

    train_loader = DataLoader(
        train_dataset, 
        batch_size = config.dataset.batch_size, 
        collate_fn = custom_collate_2d, # TODO: dynamically specify collate function
    )

    if config.training.scheduler.steps_per_epoch == -1:
        steps_per_epoch = get_steps_per_epoch(train_dataset, train_loader)
        config.training.scheduler.steps_per_epoch = steps_per_epoch

    wandb.init(
        project = config.wandb.project_name,  
        dir = config.wandb.path, 
        mode = config.wandb.mode, 
        config = config
    )

    exp_dir = wandb.run.dir
    json_path = os.path.join(exp_dir, 'results.json')

    device = torch.device(config.devices.device)
    model = Tracker(device = device, **config.model) 
    model.to(device)

    # profiler = LineProfiler(train_epoch, model, model.forward, model.forward_iteration, model.__init__)
    # profiler.enable()

    # reporter = MemReporter(model)
    # #profiler.add_function(train_epoch)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = config.training.optimizer.learning_rate, 
        weight_decay = config.training.optimizer.weight_decay
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer = optimizer,
        max_lr = config.training.optimizer.learning_rate,
        epochs = config.training.n_epochs,
        **config.training.scheduler
    )

    loss = TotalLoss(**config.training.losses)
 
    for i in range(config.training.n_epochs): 

        result_dict = {'epoch': i}

        loss_summary = train_epoch(
            model = model,
            dataloader = train_loader, 
            optimizer = optimizer,
            loss = loss, 
            debug_ix = config.training.debug_ix, 
            use_amp = config.training.use_half_precision # TODO: bug fix
        )
        result_dict.update(loss_summary)


        # if i % config.training.eval_freq == 0: 

        #     metrics_summary = eval_epoch(
        #         model = model, 
        #         dataloader = val_loader,
        #         prefix = 'val/'
        #         debug_ix = self.training.debug_ix)

        #     result_dict.update(metrics_summary)
        #     torch.cuda.empty_cache()

        # if i % config.training.checkpoint_freq == 0: 
        #     save_checkpoint(model, optimizer, criterion, 
        #         prefix = exp_dir, epoch = i)

        # log to wandb 
        wandb.log(result_dict)

        # save losses and evaluation metrics to json
        write_json(json_path, result_dict)
        wandb.save(json_path, base_path = exp_dir)

        if i % config.training.print_freq == 0:
            print(result_dict)
            
        loss.reset_history()
        # profiler.print_stats()

    wandb.finish()


if __name__ == '__main__':

    args = parse_args()
    config = load_config(args.config_path)
    main(config)