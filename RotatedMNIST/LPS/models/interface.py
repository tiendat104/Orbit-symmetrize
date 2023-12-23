import sys
import os
cwd = os.getcwd() + '/'
root_dir = os.path.join(cwd, "emlp-pytorch/") 
sys.path.append(os.path.dirname(root_dir ))

import torch 
import torch.nn as nn
from torch import Tensor as T
import torch.nn.functional as F 

import numpy as np

from emlp_pytorch.groups import *
from emlp_pytorch.reps import *
from emlp_pytorch.nn import EMLP, EMLPBlock, Linear
from emlp_pytorch.nn import uniform_rep

from .backbone import CNN
from .utils import index_to_perm
from .utils import rotate_kornia_API
from .utils import batched_gram_schmidt_2d

import time

class EquivariantInterface(nn.Module):
    def __init__(self, m = 200, ch = 96, num_layers = 3, noise_scale=1.0, dz = 10, threshold=0.1, device=None): 
        super(EquivariantInterface, self).__init__()
        self.m = m
        self.dz = dz
        self.noise_scale = noise_scale
        self.threshold = threshold
        self.bound = 1.0
        self.device = device
        
        self.get_reps(ch, num_layers)
        self.emlp_G = nn.Sequential(
            *[EMLPBlock(rin, rout) for rin, rout in zip(self.reps, self.reps[1:])],
            Linear(self.reps[-1], self.rep_out)).to(self.device)
    
    def get_reps(self, ch:int, num_layers: int):
        G = SO(2).to(self.device)
        self.G = G
        self.rep_in = self.m * T(0)(G) + (self.m + self.dz) * T(1)(G)
        rep_mids = num_layers * [uniform_rep(ch, G)]
        self.rep_out = 2*T(1)(G)
        self.reps = [self.rep_in] + rep_mids
        
    def get_coords(self): # checked
        idx1 = [i for i in range(-14, 0)] + [i for i in range(1, 15)]
        idx1 = torch.tensor(idx1, device=self.device)
        idx1 = idx1[None,:].expand(28, -1) # [28, 28]
        idx1 = idx1.unsqueeze(0) # [1, 28, 28]
        
        idx2 = [i for i in range(14, 0, -1)] + [i for i in range(-1, -15, -1)]
        idx2 = torch.tensor(idx2, device=self.device)
        idx2 = idx2[:,None].expand(-1, 28) # [28, 28]
        idx2 = idx2.unsqueeze(0) # [1, 28, 28]
        
        idx = torch.cat((idx1, idx2), dim=0) # [2, 28, 28]
        return idx
    
    def get_vec(self, images): # checked
        # images: [B, 28, 28]
        B = images.shape[0]
        coords = self.get_coords().to(self.device) # [2, 28, 28]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1) # [B, 2, 28, 28]
        
        x = torch.zeros(B, 3, 28, 28, device=self.device) # [B, 3, 28, 28]
        x[:,:2,:,:] = coords 
        x[:,2,:,:] = images
        x = torch.permute(x, (0, 2, 3, 1)) # [B, 28, 28, 3]
        x = x.reshape(B, 28*28, 3) # [B, 784, 3] 
        # checked
        
        _, indices = torch.sort(x[:,:,2], descending=True) # [B, 784]
        perms = index_to_perm(indices) # [B, 784, 784] 
        sort_x = torch.matmul(perms, x) # [B, 784, 3]
        
        v = sort_x[:, :self.m, :] # [B, m, 3]
        # mask threshold and zero out!
        mask_t = v[:,:,2] < self.threshold # [B, m]
        mask_t = mask_t[:,:,None].expand(B, self.m, 3)  # [B, m, 3]
        v[mask_t] = 0
        return v # [B, m, 3]
    
    def sample_noise(self, B): # checked
        # return Z: [B, 2, dz]
        # Uniform sampling over a unit circle
        pi = float(np.pi)
        angles = torch.zeros(B, self.dz, device=self.device).uniform_(0, 2*pi) # [B, dz]
        cos_theta = torch.cos(angles) # [B, dz]
        sin_theta = torch.sin(angles) # [B, dz]
        
        Z = torch.zeros(B, 2, self.dz, device=self.device) # [B, 2, dz] 
        Z[:,0,:] = cos_theta 
        Z[:,1,:] = sin_theta
        Z = Z * self.noise_scale
        return Z # [B, 2, dz]
    
    def formatize_input(self, v, Z):
        # v: [B, m, 3],  Z: [B, 2, dz]
        B = v.shape[0]
        coord_flat = v[:,:,:2] # [B, m, 2]
        coord_flat = coord_flat.flatten(start_dim=1) # [B, 2*m]
        Z_flat = Z.transpose(1, 2).flatten(start_dim=1) # [B, 2*dz]
        
        x = torch.zeros(B, 3*self.m + 2*self.dz).to(self.device) # [B, 3*m + 2*dz]
        x[:, :self.m] = v[:,:,2] 
        x[:, self.m : 3*self.m] = coord_flat 
        x[:, 3*self.m:] = Z_flat
        return x # [B, 3*m + 2*dz] 
    
    def postprocessing(self, gs):
        # gs: [B, 2, 2]
        B = gs.shape[0]
        # Gram-schmidt algorithm for 2d matrices 
        gs = batched_gram_schmidt_2d(gs) # O(2) matrices
        # get det
        det_gs = torch.linalg.det(gs) # [B]
        # multiple first column with det
        sign_arr = torch.ones(B, 2, device=gs.device) # [B, 2]
        sign_arr[:,0] = det_gs 
        sign_arr = sign_arr[:, None, :].expand(B, 2, 2) # [B, 2, 2]
        # elementwise multiplication
        gs = gs * sign_arr # [B, 2, 2]
        return gs 
    
    def forward(self, images):
        # images: [B, 28, 28]
        B = images.shape[0]
        
        v = self.get_vec(images) # [B, m, 3] # 0.0007
        Z = self.sample_noise(B) # [B, 2, dz]
        
        x = self.formatize_input(v, Z) # [B, 3*m + 2*dz] 
        
        gs = self.emlp_G(x) # [B, 4]  # 0.0146
        gs = gs.reshape(B, 2, 2).transpose(-1, -2) # [B, 2, 2]
        
        # SO(2)-equivariant postprocessing to get valid group elements
        gs = self.postprocessing(gs) # valid SO(2)
        # gs [B, 2, 2]
        return gs # [B, 2, 2]

class InterfacedModel(nn.Module):
    def __init__(self, seed=0, m=200, ch=96, num_layers = 3, noise_scale=1.0, dz = 10, threshold=0.1, dropout=0.4, sample_size=1, device=None):
        super().__init__()
        self.backbone = CNN(dropout=dropout, seed=seed).to(device)
        self.interface = EquivariantInterface(
            m = m,
            ch = ch,
            num_layers = num_layers,
            noise_scale = noise_scale,
            dz = dz, 
            threshold = threshold,
            device = device
        )
        self.sample_size = sample_size 
        
    def transform_input(self, images, gs):
        # images: [B, 28, 28], gs: [B, 2, 2]
        B = images.shape[0] 
        images = images.unsqueeze(1) # [B, 1, 28, 28]
        gs_inv = gs.transpose(-1, -2) # [B, 2, 2]
        images = rotate_kornia_API(images, gs_inv) # [B, 1, 28, 28]
        return images # [B, 1, 28, 28]

    def forward(self, images):
        # images: [B, 1, 28, 28]
        B = images.shape[0]
        images = images.expand(-1, self.sample_size, -1, -1) # [B, k, 28, 28]
        images = images.reshape(B*self.sample_size, 28, 28) # [B*k, 28, 28]
        
        # sample from p(g|x)
        gs = self.interface(images) # [B*k, 2, 2]
        transformed_images = self.transform_input(images, gs) # [B*k, 1, 28, 28] 
        # checked
        logits = self.backbone(transformed_images) # [B*k, 10] # 0.00142
        logits = logits.reshape(B, self.sample_size, 10) # [B, k, 10]
        logits = logits.mean(dim=1) # [B, 10]
        return logits
