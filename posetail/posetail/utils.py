import itertools
import torch

import numpy as np
import torch.nn as nn 
import torch.nn.functional as F

from aniposelib.cameras import CameraGroup
from einops import rearrange


def init_kwargs(kwargs_dict):
     
    d = {}
    if kwargs_dict is not None:
        d = kwargs_dict

    return d

def get_pos_encoding(pos, dim, dtype = torch.double): 
    ''' 
    pos: positions to encode
    dim: the embedding dimension (should match the transformer input dimension)
    '''
    assert (dim % 2 == 0)
    
    d = dim // 2
    dim_ixs = torch.arange(d, device = pos.device, dtype = dtype)

    v = 1 / (10000 ** (2 * dim_ixs / dim))
    m = torch.einsum('i,j->ij', pos, v)

    sin_encoding = torch.sin(m)
    cos_encoding = torch.cos(m)

    encoding = torch.cat((sin_encoding, cos_encoding), dim = 1)
    encoding = rearrange(encoding, 'p d -> 1 p 1 d').float()

    return encoding


def get_fourier_encoding(coords, min_freq = 0, max_freq = 1):

    freqs = 2 ** torch.arange(min_freq, max_freq, device = coords.device).float()
    freq_coords = torch.einsum('bsnr,f->bsnfr', coords, freqs)
    freq_coords = rearrange(freq_coords, 'b s n f r -> b s n (f r)')

    fourier_enc = torch.cat([torch.sin(freq_coords), torch.cos(freq_coords)], dim = -1)

    return fourier_enc


def encode_dim(enc, coords, v): 

    coords = torch.unsqueeze(coords, dim = -1)
    enc[:, :, 0::2] = torch.sin(coords * v)
    enc[:, :, 1::2] = torch.cos(coords * v)

    return enc