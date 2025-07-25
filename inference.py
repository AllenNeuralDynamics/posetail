import argparse
import os

from torch.utils.data import DataLoader

from posetail.datasets.datasets import Rat7mDataset, custom_collate_3d
from train_utils import *
from inference_utils import *
from viz3d import *


def parse_args():
    '''
    parse command line arguments
    ''' 
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-ids', nargs = '+', default = [])
    parser.add_argument('--video-paths', nargs = '+', default = [])
    parser.add_argument('--data-path')

    args = parser.parse_args()

    return args


def main(run_ids, video_paths, data_path, checkpoint = None): 

    results_outpath = safe_make('../results', exist_ok = True)
    video_outpath = safe_make('../videos', exist_ok = True)

    for run_id in run_ids:

        config_path = f'/allen/aind/scratch/katie.rupp/wandb/{run_id}/files/config.toml'
        config = load_config(config_path)

        model_path = get_checkpoint(run_id, checkpoint = checkpoint)
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

        if config.dataset.train.project_2d: 
            pred_path = os.path.join(results_outpath, f'{video_name}_{run_id}_predictions.npz')
        else: 
            video_group_name = '-'.join(video_name.split('-')[:2] + video_name.split('-')[3:])
            pred_path = os.path.join(results_outpath, f'{video_group_name}_{run_id}_predictions.npz')

        pred_path = get_video_predictions(video_paths, 
            model, dataloader, pred_path, debug_ix = -1)
            
        print(f'predictions saved to {pred_path}')

        cam = dataloader.dataset.cams[0]
        scale = dataloader.dataset.camera_size_dict[cam]['xy_scale']

        # viz for 2d 
        if config.dataset.train.project_2d: 
            video_outpath = generate_video_2d(
                video_path = video_paths[0], 
                results_path = pred_path, 
                outpath = video_path, 
                run_id = run_id, 
                scale = scale)

            print(f'video saved to {video_outpath}\n')

        # viz for 3d 
        else: 
            video_outpath = viz_predictions_3d(pred_path, results_outpath, spawn = False)    
            print(f'saved 3d predictions to {video_outpath}')


if __name__ == '__main__':

    # args = parse_args()

    # run_ids = args.run_ids
    # video_path = args.video_path
    # data_path = args.data_path

    # run-20250717_162810-2bxgzkoy
    run_ids = ['run-20250719_135301-pnscusdl', 
               'run-20250719_135329-x27vkwdb', 
               'run-20250720_050005-g7yrnvuo']
    run_ids = ['run-20250724_200413-pfcy4n9z']

    # video_paths = ['/allen/aind/scratch/katie.rupp/data/rat7m/videos/s5-d2/s5-d2-camera1-0.mp4']
    video_paths = ['/allen/aind/scratch/katie.rupp/data/rat7m/videos/s5-d2/s5-d2-camera1-0.mp4',
                    '/allen/aind/scratch/katie.rupp/data/rat7m/videos/s5-d2/s5-d2-camera2-0.mp4',
                    '/allen/aind/scratch/katie.rupp/data/rat7m/videos/s5-d2/s5-d2-camera3-0.mp4',
                    '/allen/aind/scratch/katie.rupp/data/rat7m/videos/s5-d2/s5-d2-camera4-0.mp4',
                    '/allen/aind/scratch/katie.rupp/data/rat7m/videos/s5-d2/s5-d2-camera5-0.mp4',
                    '/allen/aind/scratch/katie.rupp/data/rat7m/videos/s5-d2/s5-d2-camera6-0.mp4']
    data_path = '/allen/aind/scratch/katie.rupp/data/rat7m/data/mocap-s5-d2.mat'

    main(run_ids, video_paths, data_path, checkpoint = None)