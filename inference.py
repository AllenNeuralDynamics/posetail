import argparse
import os

from torch.utils.data import DataLoader

from posetail.datasets.inference_dataset import PosetailInferenceDataset
from posetail.datasets.posetail_dataset import custom_collate
from inference_utils import *
from viz3d import *


''' 
a utility script for running inference on a single video given a 
model checkpoint and the config file that was used to train it. 
an .rrd file will be generated to visualize the 3d predictions.

python inference.py
'''

def parse_args():
    '''
    parse command line arguments
    ''' 
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-path', type = str)
    parser.add_argument('--split', type = str, default = 'test')
    parser.add_argument('--config-path', type = str)
    parser.add_argument('--checkpoint-path', type = str)
    parser.add_argument('---outpath', type = str, default = '../output')

    args = parser.parse_args()

    return args


def run_inference(dataset_path, config_path, checkpoint_path, outpath, split = 'test'): 

    # load the config and model
    config = load_config(config_path)
    config.wandb.project_name = 'posetail-test'
    config.dataset.test.n_frames = 25
    device = (torch.device(config.devices.device) if torch.cuda.is_available() else 'cpu')
    model = load_checkpoint(config_path, checkpoint_path)
    model.eval()

    # set seed for reproducibility
    set_seeds(config.training.seed)

    # create a dataset for one video
    dataset = PosetailInferenceDataset(
        dataset_path = dataset_path, 
        config = config, 
        split = split) 

    dataloader = DataLoader(
        dataset, 
        batch_size = config.dataset.batch_size, 
        collate_fn = custom_collate,
        num_workers = config.dataset.get('num_workers', 1))

    # use the model to predict the 3d positions
    split_outpath = os.path.join(outpath, split)
    os.makedirs(split_outpath, exist_ok = True)
    predict_on_dataset_3d(model, dataloader, split_outpath, device, debug_ix = -1)

    # visualize the 3d predictions
    viz_predictions_3d(split_outpath, spawn = False)    


if __name__ == '__main__':

    # args = parse_args()

    # dataset_path = args.dataset_path
    # split = args.split
    # config_path = args.config_path
    # checkpoint_path = args.checkpoint_path
    # outpath = args.outpath

    dataset_path = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning/dex_ycb' 
    split = 'test'
    
    # pretrained on kubric
    # checkpoint_path = '/groups/karashchuk/home/karashchukl/results/posetail-pretrain/wandb/run-20260211_151402-9iwgznvx/files/checkpoints/checkpoint_599992.pth'

    # all data 
    # checkpoint_path = '/groups/karashchuk/home/karashchukl/results/posetail-finetuning/wandb/run-20260302_180456-ucj1ou1z/files/checkpoints/checkpoint_778240.pth'

    # finetuned on animal data
    checkpoint_path = '/groups/karashchuk/home/karashchukl/results/posetail-finetuning/wandb/run-20260227_111340-upq88r08/files/checkpoints/checkpoint_599992.pth'
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'config.toml')

    # outpath = '/groups/karashchuk/karashchuklab/animal-datasets-results/dex_ycb'    
    outpath = '/home/ruppk2@hhmi.org/dataset_scripts/predictions/dex_ycb'

    run_inference(dataset_path = dataset_path,
                  config_path = config_path, 
                  checkpoint_path = checkpoint_path, 
                  outpath = outpath, 
                  split = split)
    
