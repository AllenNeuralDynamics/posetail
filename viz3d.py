import argparse
import os 
import torch

import numpy as np
import rerun as rr

from einops import rearrange
from matplotlib.colors import to_rgb

from inference_utils import *
from posetail.datasets.utils import get_dirs


def format_scheme(scheme, keypoint_names):

    new_scheme = [] 
    kpt_to_ix = dict(zip(keypoint_names, range(len(keypoint_names))))

    for kpt1, kpt2 in scheme: 
        new_scheme.append([kpt_to_ix[kpt1], kpt_to_ix[kpt2]])

    return new_scheme


def viz_predictions_3d_rerun(coords_pred, coords_true, outpath = None, 
                             subject_ids = None, keypoint_names = None, scheme = None,
                             color_pred = 'xkcd:red', color_true = 'xkcd:green', 
                             color_connection = 'xkcd:sky blue', 
                             kpt_radius = 0.05, connection_radius = 0.01, 
                             connect_pred_to_gt = False, spawn = False):

    # get colors for visualization
    color_pred_rgb = to_rgb(color_pred)
    color_true_rgb = to_rgb(color_true)
    color_connection_rgb = to_rgb(color_connection)

    # TODO: convert scheme from keypoint names to index 
    if scheme is not None: 
        scheme = format_scheme(scheme, keypoint_names)
        print(scheme)

    # format subject ids if present, and reshape the coordinates
    if subject_ids is not None: 
        subject_ids = [f'_{sid}' for sid in subject_ids]
    else: 
        subject_ids = ['']

    T = coords_pred.shape[0]
    n_subjects = len(subject_ids)
    coords_pred = rearrange(coords_pred, 't (s n) r -> s t n r', s = n_subjects)
    coords_true = rearrange(coords_true, 't (s n) r -> s t n r', s = n_subjects)

    rr.init('posetail_vis_3d', spawn = spawn)

    for i in range(T):

        rr.set_time_seconds('iteration', i)

        for j in range(n_subjects): 

            # subject name
            subject = subject_ids[j]

            # get keypoints for the current frame
            kpts_true = coords_true[j, i, :, :]
            kpts_pred = coords_pred[j, i, :, :]

            # remove nans 
            kpts_true = kpts_true[torch.isfinite(kpts_true).all(dim = 1)]
            kpts_pred = kpts_pred[torch.isfinite(kpts_pred).all(dim = 1)]
            
            # log true keypoints in green
            rr.log(f'pose_true{subject}', 
                rr.Points3D(
                    kpts_true, 
                    colors = color_true_rgb,
                    radii = kpt_radius
                )
            )

            # log predicted keypoints in red
            rr.log(f'pose_pred{subject}', 
                rr.Points3D(
                    kpts_pred, 
                    colors = color_pred_rgb,
                    radii = kpt_radius
                )
            )
        
            # log connections between ground truth points and 
            # the corresponding predictions
            if connect_pred_to_gt:

                for j in range(kpts_pred.shape[0]):
                    rr.log(
                        f'connections{subject}/point_{j}', # NOTE: depending on nans, this kpt number may not correspond across frames
                        rr.LineStrips3D(
                            strips = [[kpts_pred[j], kpts_true[j]]],
                            colors = color_connection_rgb,
                            radii = connection_radius
                        )
                    )

            # log connections between coords if given a pose skeleton
            valid_connections_true = []
            valid_connections_pred = []

            if scheme: 

                for start_ix, end_ix in scheme:

                    # save valid connections between keypoints in ground truth
                    if not any(torch.isnan(kpts_true[start_ix])) and not any(torch.isnan(kpts_true[end_ix])):

                        valid_connections_true.append([
                            kpts_true[start_ix, :],
                            kpts_true[end_ix, :]
                        ])

                    # save valid connections between keypoints in model predictions
                    if not any(torch.isnan(kpts_pred[start_ix])) and not any(torch.isnan(kpts_pred[end_ix])):

                        valid_connections_pred.append([
                            kpts_pred[start_ix, :],
                            kpts_pred[end_ix, :]
                        ])

                # log valid skeleton between true keypoints in green
                if valid_connections_true:

                    rr.log(f'pose/connections_true{subject}',
                        rr.LineStrips3D(
                            strips = valid_connections_true,
                            colors = color_true_rgb, 
                            radii = connection_radius
                        )
                    )

                # log valid skeleton between predicted keypoints in red
                if valid_connections_pred:

                    rr.log(f'pose/connections_pred{subject}',
                        rr.LineStrips3D(
                            strips = valid_connections_pred,
                            colors = color_pred_rgb, 
                            radii = connection_radius
                        )
                    )

    if outpath: 
        rr.save(outpath)
    
    return outpath


def viz_predictions_3d(split_path, spawn = False, **kwargs):

    device = torch.device('cpu')

    for session in get_dirs(split_path): 

        session_path = os.path.join(split_path, session)

        for trial in get_dirs(session_path):

            trial_path = os.path.join(session_path, trial)

            # skip if there are no npz prediction files
            predictions_path = os.path.join(trial_path, 'predictions.npz')
            if not os.path.exists(predictions_path): 
                print(f'skipping... no predictions found at {trial_path}')
                continue

            # load the coords 
            data = np.load(predictions_path)
            coords_true = torch.from_numpy(data['coords_true'])
            coords_pred = torch.from_numpy(data['coords_pred'])

            subject_ids = None
            if 'subject_ids' in data:
                subject_ids = data['subject_ids']

            # get keypoint names and scheme (if present) from the trial path
            pose_path = data['pose_path']
            scheme = None 
            keypoint_names = None

            if os.path.exists(pose_path): 

                pose_data = np.load(pose_path)
                keypoint_names = pose_data['keypoints']

                if 'scheme' in pose_data: 
                    scheme = pose_data['scheme']

            # save to rrd file (to visualize with rerun)
            rrd_outpath = os.path.join(trial_path, f'predictions_3d.rrd')
            rrd_outpath = viz_predictions_3d_rerun(
                coords_pred, coords_true, rrd_outpath, 
                subject_ids = subject_ids, scheme = scheme,
                keypoint_names = keypoint_names, spawn = spawn,
                **kwargs)
            print(f'saved 3d predictions to {rrd_outpath}')