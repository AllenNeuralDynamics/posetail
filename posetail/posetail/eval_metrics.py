import torch
import numpy as np

def get_eval_metrics(vis_pred, vis_true, coords_pred, 
                     coords_true, thresholds, prefix = 'eval/'):

    occlusion_acc = get_occlusion_accuracy(vis_pred, vis_true)

    delta_x_avg, delta_xs = get_delta_x_avg(coords_pred, coords_true, 
                    vis_pred, vis_true, thresholds)

    metrics = {f'{prefix}occlusion_acc': occlusion_acc, 
               f'{prefix}delta_x_avg': delta_x_avg}

    for i, thresh in enumerate(thresholds): 
        metrics[f'{prefix}delta_x_{thresh}'] = delta_xs[i]

    return metrics


def get_occlusion_accuracy(vis_pred, vis_true): 
    ''' 
    parameters:
        vis_pred: B, T, N, 1
        vis_true: B, T, N, 1

    returns: 
        occlusion_acc (float)
    '''
    occlusion_pred = ~vis_pred
    occlusion_true = ~vis_true

    occlusion_acc = torch.sum(vis_pred == vis_true) / torch.numel(vis_pred)

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
    within_thresh = torch.sum(coords_pred ** 2 - coords_true ** 2, dim = -1) < (thresh ** 2)
    good = within_thresh and vis_true
    delta_x = torch.sum(good, dim = (0, 1, 2)) / torch.sum(vis_true)

    return delta_x 


def get_delta_x_avg(coords_pred, coords_true, 
                    vis_pred, vis_true, thresholds = None): 

    delta_xs = []
    
    # initialize to default values
    if thresholds is None: 
        thresholds = [1, 2, 4, 8, 16]

    for thresh in thresholds:

        delta_x = get_delta_x(
            coords_pred, coords_true, 
            vis_pred, vis_true, 
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

    valid_mask = (vis_pred and vis_true).squeeze(-1)

    error_per_kpt = torch.norm(coords_pred - coords_true, dim = -1)
    error = torch.nansum(error, dim = -1) / (valid_mask.sum(dim = -1) + eps)
    
    mask = valid_mask.sum(dim = -1) == 0
    error[mask] = float('nan')

    avg_error = torch.nanmean(error)

    return mpjpe