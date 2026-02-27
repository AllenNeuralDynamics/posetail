import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from posetail.posetail.cube import get_camera_scale


class TotalLoss(nn.Module): 

    def __init__(self, gamma = 0.8, pixel_thresh = 12, delta = 6, 
                 use_huber_loss = False, vis_loss_weight = 1, 
                 conf_loss_weight = 1, coords_loss_weight = 1, 
                 occluded_coords_loss_weight = 1, 
                 feature_loss_weight = 0.5):
        super().__init__()

        self.gamma = gamma
        self.pixel_thresh = pixel_thresh
        self.delta = delta
        
        self.use_huber_loss = use_huber_loss

        # weight for each loss (0 to not use or compute)
        self.vis_loss_weight = vis_loss_weight
        self.conf_loss_weight = conf_loss_weight
        self.coords_loss_weight = coords_loss_weight
        self.occluded_coords_loss_weight = occluded_coords_loss_weight
        self.feature_loss_weight = feature_loss_weight

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
            use_huber_loss = self.use_huber_loss, 
            weight = self.coords_loss_weight
        )

        self.mae_loss_occluded_coords = WeightedMAELoss(
            gamma = self.gamma, 
            delta = self.delta, 
            use_huber_loss = self.use_huber_loss, 
            weight = self.occluded_coords_loss_weight
        )

        if self.feature_loss_weight > 0: 
            self.feature_loss = FeatureLoss(
                weight = self.feature_loss_weight)

        loss_names = ['vis_loss', 'conf_loss', 
                      'occluded_coords_loss', 'coords_loss',
                      'feature_loss','bad_feature_loss',
                      'total_loss', 'cube_scale']
        self.loss_history = {loss_name: [] for loss_name in loss_names}

    def collapse_history(self, prefix = ''): 
        
        loss_summary = {}

        for name, losses in self.loss_history.items():
            loss_summary[f'{prefix}{name}'] = float(np.nanmean(losses))
            
        return loss_summary

    def reset_history(self):
        self.loss_history = {name: [] for name in list(self.loss_history.keys())}


    def forward(self, model, outputs, coords_true, 
                vis_true, cgroup = None, device = None):

        coords_pred = outputs['coords_pred']
        vis_pred = outputs['vis_pred']
        conf_pred = outputs['conf_pred']
        occluded_true = ~vis_true

        if model.training:

            coords_pred_iters = outputs['coords_pred_iters']
            vis_pred_iters = outputs['vis_pred_iters']
            conf_pred_iters = outputs['conf_pred_iters']

            coords_true_unrolled, vis_true_unrolled, occluded_true_unrolled = unroll_batch(
                coords = coords_true, 
                vis = vis_true, 
                stride = model.S, 
                stride_overlap = model.stride_overlap)

        # compute losses
        vis_loss = self.bce_loss_vis(
            vis_pred = vis_pred_iters if model.training else vis_pred,
            vis_true = vis_true_unrolled if model.training else vis_true,
            device = device
        )

        conf_loss = self.bce_loss_conf(
            conf_pred = conf_pred_iters if model.training else conf_pred, 
            coords_pred = coords_pred_iters if model.training else coords_pred,  
            coords_true = coords_true_unrolled if model.training else coords_true, 
            vis_true = vis_true_unrolled if model.training else vis_true,
            device = device
        )

        coords_loss = self.mae_loss_coords(
            coords_pred = coords_pred_iters if model.training else coords_pred, 
            coords_true = coords_true_unrolled if model.training else coords_true, 
            vis_true = vis_true_unrolled if model.training else vis_true, 
            device = device
        )

        occluded_coords_loss = self.mae_loss_occluded_coords(
            coords_pred = coords_pred_iters if model.training else coords_pred, 
            coords_true = coords_true_unrolled if model.training else coords_true, 
            vis_true = occluded_true_unrolled if model.training else occluded_true, 
            device = device
        )

        feature_loss, bad_feature_loss = self.feature_loss(
                model = model, 
                coords_true = coords_true, 
                feature_planes_levels = outputs['feature_planes_levels'], 
                cgroup = cgroup,
                device = device
        )

        if model.R == 3:
            scale = get_camera_scale(cgroup, coords_true.reshape(-1, 3))
            coords_loss = coords_loss / scale
            occluded_coords_loss = occluded_coords_loss / scale
        else:
            scale = 1
            
        losses = [coords_loss, occluded_coords_loss, 
                  vis_loss, conf_loss, 
                  feature_loss, bad_feature_loss]
        
        # total_loss = 0
        # for loss in losses: 
        #     if torch.isfinite(loss).item(): 
        #         total_loss += loss
        losses = torch.stack(losses)
        losses = losses[torch.isfinite(losses)]
        total_loss = losses.sum()

        self.loss_history['coords_loss'].append(coords_loss.item())
        self.loss_history['occluded_coords_loss'].append(occluded_coords_loss.item())
        self.loss_history['vis_loss'].append(vis_loss.item())
        self.loss_history['conf_loss'].append(conf_loss.item())
        self.loss_history['total_loss'].append(total_loss.item())
        self.loss_history['feature_loss'].append(feature_loss.item())
        self.loss_history['bad_feature_loss'].append(bad_feature_loss.item())

        self.loss_history['cube_scale'].append(scale)
        
        return total_loss


class BCELossVis(nn.Module): 
    
    def __init__(self, gamma = 0.8, weight = 1):
        super().__init__()

        self.gamma = gamma 
        self.weight = weight

    def _compute_loss(self, vis_pred, vis_true): 

        loss = F.binary_cross_entropy_with_logits(
            vis_pred, 
            vis_true.float(), 
            reduction = 'mean')

        return loss 

    def forward(self, vis_pred, vis_true, device = None):

        # don't compute if the weight is 0
        if self.weight == 0: 
            return torch.tensor(float('nan'), device = device)

        if isinstance(vis_pred, torch.Tensor): 
            total_loss = self._compute_loss(vis_pred, vis_true)
            return self.weight * total_loss 

        n_strides = len(vis_pred)
        n_iters = len(vis_pred[0])

        losses = torch.ones((n_strides, n_iters), device = device)
        weights = self.gamma ** torch.arange(n_iters, device = device).flip(0)

        for i in range(n_strides):
            for j in range(n_iters):
                losses[i, j] = self._compute_loss(vis_pred[i][j], vis_true[i])

        total_loss = self.weight * torch.nanmean(weights * torch.nanmean(losses, axis = 0), axis = 0)

        return total_loss


class BCELossConf(nn.Module): 

    def __init__(self, gamma = 0.8, pixel_thresh = 12, weight = 1): 
        super().__init__()

        self.gamma = gamma 
        self.pixel_thresh = pixel_thresh
        self.weight = weight

    def _compute_loss(self, conf_pred, coords_pred, coords_true, vis_true): 

        dist = torch.sum((coords_pred - coords_true) ** 2, dim = -1) ** 0.5
        mask = (dist <= self.pixel_thresh).float().unsqueeze(dim = -1)

        loss = F.binary_cross_entropy_with_logits(
            conf_pred, 
            mask, 
            reduction = 'mean')

        loss = torch.nanmean(loss * vis_true)

        return loss 

    def forward(self, conf_pred, coords_pred, coords_true, vis_true, device = None): 

        # don't compute if the weight is 0
        if self.weight == 0: 
            return torch.tensor(float('nan'), device = device)
 
        if isinstance(coords_pred, torch.Tensor): 
            total_loss = self._compute_loss(conf_pred, coords_pred, coords_true, vis_true)
            return self.weight * total_loss 

        n_strides = len(conf_pred)
        n_iters = len(conf_pred[0])

        losses = torch.ones((n_strides, n_iters), device = device)
        weights = self.gamma ** torch.arange(n_iters, device = device).flip(0)

        for i in range(n_strides): 
            for j in range(n_iters):
                losses[i, j] = self._compute_loss(
                    conf_pred[i][j], 
                    coords_pred[i][j], 
                    coords_true[i], 
                    vis_true[i])

        total_loss = self.weight * torch.nanmean(weights * torch.nanmean(losses, axis = 0), axis = 0)

        return total_loss 

class WeightedMAELoss(nn.Module):

    def __init__(self, gamma = 0.8, delta = 6, use_huber_loss = False, weight = 1):
        super().__init__()

        self.gamma = gamma
        self.delta = delta
        self.use_huber_loss = use_huber_loss
        self.weight = weight

    def huber_loss(self, coords_pred, coords_true):

        diff = coords_pred - coords_true
        mask = torch.abs(diff) <= self.delta
        
        loss_masked = 0.5 * ((diff * mask) ** 2) 
        loss_unmasked = ~mask * self.delta * (torch.abs(diff * ~mask) - 0.5 * self.delta)

        total_loss = loss_masked + loss_unmasked

        return total_loss
    
    def _compute_loss(self, coords_pred, coords_true, vis_true): 

        if self.use_huber_loss: 
            loss = self.huber_loss(coords_pred, coords_true)
        else:
            loss = torch.abs(coords_pred - coords_true)

        loss = torch.nanmean(loss * vis_true)
        
        return loss 

    def forward(self, coords_pred, coords_true, vis_true, device = None): 

        # don't compute if the weight is 0
        if self.weight == 0: 
            return torch.tensor(float('nan'), device = device)

        if isinstance(coords_pred, torch.Tensor): 
            total_loss = self._compute_loss(coords_pred, coords_true, vis_true)
            return self.weight * total_loss 

        n_strides = len(coords_pred)
        n_iters = len(coords_pred[0])

        losses = torch.ones((n_strides, n_iters), device = device)
        weights = self.gamma ** torch.arange(n_iters, device = device).flip(0)

        for i in range(n_strides):
            for j in range(n_iters):
                losses[i, j] = self._compute_loss(coords_pred[i][j], coords_true[i], vis_true[i])

        total_loss = self.weight * torch.nanmean(weights * torch.nanmean(losses, axis = 0), axis = 0)

        return total_loss


class FeatureLoss(nn.Module): 

    def __init__(self, weight):
        super().__init__()

        self.weight = weight

    def forward(self, model, coords_true, feature_planes_levels, cgroup, device = None): 

        # don't compute if the weight is 0
        if self.weight == 0: 
            feature_loss = torch.tensor(float('nan'), device = device)
            bad_feature_loss = torch.tensor(float('nan'), device = device)
            return feature_loss, bad_feature_loss

        feature_loss = model.get_feature_loss(
            feature_planes_levels = feature_planes_levels, 
            coords_full = coords_true, 
            camera_group = cgroup)
        
        b, s, n, r = coords_true.shape
        coords_flat = rearrange(coords_true, 'b s n r -> (b s n) r')
        ixs_perm = torch.randperm(coords_flat.shape[0])
        coords_shuffle = rearrange(coords_flat[ixs_perm], '(b s n) r -> b s n r',
            b = b, s = s, n = n)

        bad_feature_loss = model.get_feature_loss(
            feature_planes_levels = feature_planes_levels, 
            coords_full = coords_shuffle, 
            camera_group = cgroup)
        bad_feature_loss  = 1 - bad_feature_loss
        
        feature_loss *= self.weight
        bad_feature_loss *= self.weight
    
        return feature_loss, bad_feature_loss


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
    occluded_unrolled = []

    for i in range(n_windows): 

        ix = stride_remainder * i
        coords_subset = coords[:, ix:ix + stride, ...]
        vis_subset = vis[:, ix:ix + stride, ...]

        coords_unrolled.append(coords_subset)
        vis_unrolled.append(vis_subset)
        occluded_unrolled.append(~vis_subset)

    return coords_unrolled, vis_unrolled, occluded_unrolled

