import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class TotalLoss(nn.Module): 

    def __init__(self, gamma = 0.8, pixel_thresh = 12, delta = 6, 
                 use_huber = False, vis_loss_weight = 1,
                 conf_loss_weight = 1, coords_loss_weight = 1):
        super().__init__()

        self.gamma = gamma
        self.pixel_thresh = pixel_thresh
        self.delta = delta
        self.use_huber = use_huber

        self.vis_loss_weight = vis_loss_weight
        self.conf_loss_weight = conf_loss_weight
        self.coords_loss_weight = coords_loss_weight

        self.bce_loss_vis = BCELossVis(
            gamma = self.gamma, 
            weight = self.vis_loss_weight
        )

        self.bce_loss_conf = BCELossConf(
            gamma = self.gamma, 
            pixel_thresh = self.pixel_thresh, 
            weight = self.conf_loss_weight
        )

        self.mae_loss_coords = WeightedMAELoss(
            gamma = self.gamma, 
            delta = self.delta, 
            use_huber = self.use_huber, 
            weight = self.coords_loss_weight
        )

        loss_names = ['vis_loss', 'conf_loss', 'coords_loss', 'feature_loss', 'bad_feature_loss', 'total_loss']
        self.loss_history = {loss_name: [] for loss_name in loss_names}

    def collapse_history(self, prefix = ''): 
        
        loss_summary = {}

        for name, losses in self.loss_history.items():
            loss_summary[f'{prefix}{name}'] = float(np.nanmean(losses))
            
        return loss_summary

    def reset_history(self):
        self.loss_history = {name: [] for name in list(self.loss_history.keys())}


    def forward(self, outputs, coords_true, vis_true, device = None):

        (coords_pred, vis_pred, 
        conf_pred, coords_pred_iters, 
        vis_pred_iters, conf_pred_iters) = outputs[:6]

        # compute losses
        vis_loss = self.bce_loss_vis(
            vis_pred = vis_pred_iters,
            vis_true = vis_true,
            device = device
        )

        conf_loss = self.bce_loss_conf(
            conf_pred = conf_pred_iters, 
            coords_pred = coords_pred_iters, 
            coords_true = coords_true, 
            vis_true = vis_true,
            device = device
        )

        coords_loss = self.mae_loss_coords(
            coords_pred = coords_pred_iters, 
            coords_true = coords_true, 
            vis_true = vis_true, 
            device = device
        )

        feature_loss = outputs[6] * 0.5
        bad_feature_loss = outputs[7] * 0.5

        # total_loss = vis_loss + conf_loss + coords_loss
        # total_loss = coords_loss
        total_loss = coords_loss + feature_loss + bad_feature_loss

        self.loss_history['vis_loss'].append(vis_loss.item())
        self.loss_history['conf_loss'].append(conf_loss.item())
        self.loss_history['coords_loss'].append(coords_loss.item())
        self.loss_history['total_loss'].append(total_loss.item())
        self.loss_history['feature_loss'].append(feature_loss.item())
        self.loss_history['bad_feature_loss'].append(bad_feature_loss.item())

        return total_loss


class BCELossVis(nn.Module): 
    
    def __init__(self, gamma = 0.8, weight = 1):
        super().__init__()

        self.gamma = gamma 
        self.weight = weight

    def forward(self, vis_pred, vis_true, device = None):

        n_strides = len(vis_pred)
        n_iters = len(vis_pred[0])

        losses = torch.ones((n_strides, n_iters), device = device)
        weights = self.gamma ** torch.arange(n_iters, device = device).flip(0)

        for i in range(n_strides):
            for j in range(n_iters):

                losses[i, j] = F.binary_cross_entropy_with_logits(
                    vis_pred[i][j], 
                    vis_true[i].float(), 
                    reduction = 'mean'
                )

        total_loss = self.weight * torch.mean(weights * torch.mean(losses, axis = 0), axis = 0)

        return total_loss


class BCELossConf(nn.Module): 

    def __init__(self, gamma = 0.8, pixel_thresh = 12, weight = 1): 
        super().__init__()

        self.gamma = gamma 
        self.pixel_thresh = pixel_thresh
        self.weight = weight

    def forward(self, conf_pred, coords_pred, coords_true, vis_true, device = None): 

        n_strides = len(conf_pred)
        n_iters = len(conf_pred[0])

        losses = torch.ones((n_strides, n_iters), device = device)
        weights = self.gamma ** torch.arange(n_iters, device = device).flip(0)

        for i in range(n_strides): 
            for j in range(n_iters):

                dist = torch.sum((coords_pred[i][j] - coords_true[i]) ** 2, dim = -1) ** 0.5
                mask = (dist <= self.pixel_thresh).float().unsqueeze(dim = -1)

                loss = F.binary_cross_entropy_with_logits(
                    conf_pred[i][j], 
                    mask, 
                    reduction = 'mean'
                )

                losses[i, j] = torch.mean(loss * vis_true[i])

        total_loss = self.weight * torch.mean(weights * torch.mean(losses, axis = 0), axis = 0)

        return total_loss 

class WeightedMAELoss(nn.Module):

    def __init__(self, gamma = 0.8, delta = 6, use_huber = False, weight = 1):
        super().__init__()

        self.gamma = gamma
        self.delta = delta
        self.use_huber = use_huber
        self.weight = weight

    def huber_loss(self, coords_pred, coords_true):

        diff = coords_pred - coords_true
        mask = torch.abs(diff) <= self.delta
        
        loss_masked = 0.5 * ((diff * mask) ** 2) 
        loss_unmasked = ~mask * self.delta * (torch.abs(diff * ~mask) - 0.5 * self.delta)

        total_loss = loss_masked + loss_unmasked

        return total_loss

    def forward(self, coords_pred, coords_true, vis_true, device = None): 

        n_strides = len(coords_pred)
        n_iters = len(coords_pred[0])

        losses = torch.ones((n_strides, n_iters), device = device)
        weights = self.gamma ** torch.arange(n_iters, device = device).flip(0)

        for i in range(n_strides):
            for j in range(n_iters):

                if self.use_huber: 
                    loss = self.huber_loss(coords_pred[i][j], coords_true[i])
                else:
                    loss = torch.abs(coords_pred[i][j] - coords_true[i])

                losses[i, j] = torch.mean(loss * vis_true[i])

        total_loss = self.weight * torch.mean(weights * torch.mean(losses, axis = 0), axis = 0)

        return total_loss


def get_vis_true(coords):

    vis = ~torch.isnan(torch.einsum('bsnr->bsn', coords))
    vis = rearrange(vis, 'b s n -> b s n 1')

    return vis 


def unroll_batch(coords, vis, stride = 8, stride_overlap = 4): 

    T = coords.shape[1]
    stride_remainder = stride - stride_overlap
    n_windows = T // (stride_remainder)
    
    coords_unrolled = []
    vis_unrolled = []

    for i in range(n_windows): 

        ix = stride_remainder * i
        coords_subset = coords[:, ix:ix + stride, ...]
        vis_subset = vis[:, ix:ix + stride, ...]

        coords_unrolled.append(coords_subset)
        vis_unrolled.append(vis_subset)

    return coords_unrolled, vis_unrolled

