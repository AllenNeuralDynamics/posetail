import argparse
import os 
import torch

import numpy as np
import rerun as rr

from matplotlib.colors import to_rgb

from inference_utils import *
from posetail.datasets.utils import get_dirs


# SKELETON_INDICES = [
#     (0, 1),    # HeadF to HeadB
#     (0, 2),    # HeadF to HeadL
#     (1, 3),    # HeadB to SpineF
#     (2, 3),    # HeadL to SpineF
#     (3, 4),    # SpineF to SpineM
#     (4, 5),    # SpineM to SpineL
#     (5, 8),    # SpineL to HipL
#     (5, 9),    # SpineL to HipR
#     (12, 10),  # ShoulderL to ElbowL
#     (10, 11),  # ElbowL to ArmL
#     (13, 14),  # ShoulderR to ElbowR
#     (14, 15),  # ElbowR to ArmR
#     (3, 12),   # SpineF to ShoulderL
#     (3, 13),   # SpineF to ShoulderR
#     (8, 17),   # HipL to KneeL
#     (17, 18),  # KneeL to ShinL
#     (9, 16),   # HipR to KneeR
#     (16, 19)   # KneeR to ShinR
# ]


def viz_predictions_3d_rerun(coords_pred, coords_true, outpath = None, skeleton_indices = None,
                             color_pred = 'xkcd:red', color_true = 'xkcd:green', 
                             color_connection = 'xkcd:sky blue', 
                             kpt_radius = 0.05, connection_radius = 0.01, 
                             connect_pred_to_gt = False, spawn = False):

    color_pred_rgb = to_rgb(color_pred)
    color_true_rgb = to_rgb(color_true)
    color_connection_rgb = to_rgb(color_connection)

    T, _, _ = coords_pred.shape
    rr.init('posetail_vis_3d', spawn = spawn)

    for i in range(T):

        rr.set_time_seconds('iteration', i)

        # get keypoints for the current frame
        kpts_true = coords_true[i, :, :]
        kpts_pred = coords_pred[i, :, :]

        # remove nans 
        kpts_true = kpts_true[torch.isfinite(kpts_true).all(dim = 1)]
        kpts_pred = kpts_pred[torch.isfinite(kpts_pred).all(dim = 1)]
        
        # log true keypoints in green
        rr.log('pose_true', 
            rr.Points3D(
                kpts_true, 
                colors = color_true_rgb,
                radii = kpt_radius
            )
        )

        # log predicted keypoints in red
        rr.log('pose_pred', 
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
                    f'connections/point_{j}', # NOTE: depending on nans, this kpt number may not correspond across frames
                    rr.LineStrips3D(
                        strips = [[kpts_pred[j], kpts_true[j]]],
                        colors = color_connection_rgb,
                        radii = connection_radius
                    )
                )

        # log connections between coords if given a pose skeleton
        valid_connections_true = []
        valid_connections_pred = []

        if skeleton_indices: 

            for start_ix, end_ix in skeleton_indices:

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

                rr.log('pose/connections_true',
                    rr.LineStrips3D(
                        strips = valid_connections_true,
                        colors = color_true_rgb, 
                        radii = connection_radius
                    )
                )

            # log valid skeleton between predicted keypoints in red
            if valid_connections_pred:

                rr.log('pose/connections_pred',
                    rr.LineStrips3D(
                        strips = valid_connections_pred,
                        colors = color_pred_rgb, 
                        radii = connection_radius
                    )
                )

    if outpath: 
        rr.save(outpath)
    
    return outpath


def viz_predictions_3d(split_path, spawn = False):

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

            data = np.load(predictions_path)
            coords_true = torch.from_numpy(data['coords_true'])
            coords_pred = torch.from_numpy(data['coords_pred'])

            # save to rrd file (to visualize with rerun)
            rrd_outpath = os.path.join(trial_path, f'predictions_3d.rrd')
            rrd_outpath = viz_predictions_3d_rerun(coords_pred, coords_true, rrd_outpath, spawn = spawn)
            print(f'saved 3d predictions to {rrd_outpath}')