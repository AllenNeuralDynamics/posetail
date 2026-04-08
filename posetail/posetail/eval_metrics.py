import torch
import numpy as np
from posetail.posetail.losses import get_vis_true

def get_eval_metrics(vis_pred, vis_true, coords_pred, 
                     coords_true, thresholds = None, prefix = 'eval/'):

    if vis_true is None:
        vis_true = get_vis_true(coords_true) 
    
    vis_pred = vis_pred.detach().cpu().to(torch.float32).numpy()
    vis_true = vis_true.detach().cpu().numpy()
    coords_pred = coords_pred.detach().cpu().to(torch.float32).numpy()
    coords_true = coords_true.detach().cpu().to(torch.float32).numpy()

    if thresholds is None:
        thresholds = [1, 2, 4, 8, 16]

    occlusion_acc = get_occlusion_accuracy(vis_pred, vis_true)

    joint_error = get_mpjpe(coords_pred, coords_true, vis_pred, vis_true)

    delta_x_avg, delta_x_dict = get_delta_x_avg(coords_pred, 
        coords_true, vis_true, thresholds = thresholds)

    metrics = {f'{prefix}occlusion_acc': occlusion_acc,
               f'{prefix}mpjpe': joint_error,
               f'{prefix}delta_x_avg': delta_x_avg}

    for k, v in delta_x_dict.items(): 
        metrics[f'{prefix}delta_x_{k}'] = v

    return metrics


def get_occlusion_accuracy(vis_pred, vis_true): 
    ''' 
    parameters:
        vis_pred: B, T, N, 1
        vis_true: B, T, N, 1

    returns: 
        occlusion_acc (float)
    '''

    occlusion_pred = vis_pred < 0.5
    occlusion_true = ~vis_true

    occlusion_acc = np.mean(occlusion_pred == occlusion_true)

    return occlusion_acc


def get_delta_x(coords_pred, coords_true, vis_true, threshold):
    ''' 
    for points that are visible, measures the fraction of 
    points that are within a distance delta pixels from 
    their ground truth

    parameters: 
        coords_pred: B, T, N, 3
        coords_true: B, T, N, 3
        vis_true: B, T, N, 1

    ''' 

    within_thresh = np.sum(coords_pred ** 2 - coords_true ** 2, axis = -1) < (threshold ** 2)
    good = within_thresh[..., None] & vis_true
    delta_x = np.sum(good, axis = (0, 1, 2)) / np.sum(vis_true)

    return delta_x 


def get_delta_x_avg(coords_pred, coords_true, 
                    vis_true, thresholds = None): 

    delta_xs = []
    
    # initialize to default values
    if thresholds is None: 
        thresholds = [1, 2, 4, 8, 16]

    for thresh in thresholds:

        delta_x = get_delta_x(
            coords_pred = coords_pred, 
            coords_true = coords_true, 
            vis_true = vis_true, 
            threshold = thresh)

        delta_xs.append(delta_x)

    delta_x_avg = np.mean(delta_xs)
    delta_x_dict = dict(zip(thresholds, delta_xs))

    return delta_x_avg, delta_x_dict 


def get_mpjpe(coords_pred, coords_true, vis_pred, vis_true, eps = 1e-8):
    ''' 
    calculates the mean per joint position error for all 
    keypoints (pixels for 2d, mm for 3d) and timepoints
    in a batch

    parameters: 
        coords_pred: B, T, N, 3
        coords_true: B, T, N, 3
        vis_pred: B, T, N, 1
        vis_true: B, T, N, 1
        eps: a small constant to prevent divide by zero errors
    '''

    # mask = (vis_pred > 0.5) & vis_true
    mask = vis_true
    valid_mask = np.squeeze(mask, axis = -1)

    error_per_kpt = np.linalg.norm(coords_pred - coords_true, axis = -1, keepdims = False)
    error = np.nansum(error_per_kpt, axis = -1) / (np.sum(valid_mask, axis = -1) + eps)
    
    mask = np.sum(valid_mask, axis = -1) == 0
    error[mask] = np.nan

    mpjpe = np.nanmean(error)

    return mpjpe

