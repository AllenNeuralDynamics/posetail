import argparse
import os

from torch.utils.data import DataLoader

from posetail.datasets.datasets import Rat7mDataset, custom_collate_3d
from train_utils import *
from inference_utils import *
from viz3d import *

''' 
python inference.py
'''

def parse_args():
    '''
    parse command line arguments
    ''' 
    parser = argparse.ArgumentParser()

    parser.add_argument('--wandb-prefix', type = str)
    parser.add_argument('--run-ids', nargs = '+', default = [])

    parser.add_argument('--video-paths', nargs = '+', default = [])
    parser.add_argument('--data-path', type = str)

    parser.add_argument('---outpath', type = str, default = '../output')

    args = parser.parse_args()

    return args


def main(wandb_prefix, run_ids, video_paths, data_path, outpath, checkpoint = None): 

    # outpath = safe_make(outpath, exist_ok = True)
    outpath = safe_make(os.path.join(outpath, 'results'), exist_ok = True)

    for run_id in run_ids:

        config_path = os.path.join(wandb_prefix, run_id, 'files', 'config.toml')
        config = load_config(config_path)
        device = (torch.device(config.devices.device) if torch.cuda.is_available() else 'cpu')

        model_path = get_checkpoint(wandb_prefix, run_id, checkpoint = checkpoint)
        print(f'loading: {model_path}...')
        model = load_checkpoint(config_path, model_path)
        model.eval()

        set_seeds(config.training.seed)

        dataset = Rat7mDataset(
            video_paths = video_paths, 
            data_path = data_path, 
            n_frames = config.dataset.test.n_frames, 
            max_res = config.dataset.train.max_res) # TODO: add to config and change to test

        dataloader = DataLoader(
            dataset, 
            batch_size = config.dataset.batch_size, 
            collate_fn = custom_collate_3d)

        video_name = os.path.splitext(os.path.basename(video_paths[0]))[0]

        if config.dataset.test.project_2d: 
            video_outpath = safe_make(os.path.join(outpath, 'videos'), exist_ok = True)
            results_outpath = safe_make(os.path.join(outpath, 'results'), exist_ok = True)
            pred_path = os.path.join(results_outpath, f'{video_name}_{run_id}_predictions.npz')

        else: 
            results_outpath = safe_make(os.path.join(outpath, 'results'), exist_ok = True)
            video_group_name = '-'.join(video_name.split('-')[:2] + video_name.split('-')[3:])
            pred_path = os.path.join(results_outpath, f'{video_group_name}_{run_id}_predictions.npz')

        pred_path = get_video_predictions(video_paths, 
            model, dataloader, pred_path, device, debug_ix = -1)
            
        print(f'predictions saved to {pred_path}')

        cam = dataloader.dataset.cams[0]
        scale = dataloader.dataset.camera_size_dict[cam]['xy_scale']

        # viz for 2d 
        if config.dataset.test.project_2d:

            video_outpath = generate_video_2d(
                video_path = video_paths[0], 
                results_path = pred_path, 
                outpath = results_outpath, 
                run_id = run_id, 
                scale = scale,
                device = device)

            print(f'video saved to {video_outpath}\n')

        # viz for 3d 
        else: 
            rrd_outpath = viz_predictions_3d(pred_path, results_outpath, spawn = False)    
            print(f'saved 3d predictions to {rrd_outpath}')


if __name__ == '__main__':

    # args = parse_args()

    # wandb_prefix = args.wandb_prefix
    # run_ids = args.run_ids
    # video_path = args.video_path
    # data_path = args.data_path
    # outpath = args.outpath

    outpath = '/home/ruppk2@hhmi.org/output'
    wandb_prefix = '/data/results/katie/wandb'

    run_ids = ['run-20251119_005700-d6nqqj3q']

    video_paths = ['/data/animal-datasets/rat7m/videos/s1-d1/s1-d1-camera1-3500.mp4',
                    '/data/animal-datasets/rat7m/videos/s1-d1/s1-d1-camera2-3500.mp4',
                    '/data/animal-datasets/rat7m/videos/s1-d1/s1-d1-camera3-3500.mp4',
                    '/data/animal-datasets/rat7m/videos/s1-d1/s1-d1-camera4-3500.mp4',
                    '/data/animal-datasets/rat7m/videos/s1-d1/s1-d1-camera5-3500.mp4',
                    '/data/animal-datasets/rat7m/videos/s1-d1/s1-d1-camera6-3500.mp4']
    data_path = '/data/animal-datasets/rat7m/data/mocap-s1-d1.mat'


    main(wandb_prefix = wandb_prefix, 
         run_ids = run_ids, 
         video_paths = video_paths, 
         data_path = data_path, 
         outpath = outpath, 
         checkpoint = None)