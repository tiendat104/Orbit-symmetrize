import sys
import os
cwd = os.getcwd() + '/'
root_dir = os.path.join(cwd, "emlp-pytorch/") 
sys.path.append(os.path.dirname(root_dir ))

import torch 
import torch.nn as nn
import numpy as np

from emlp_pytorch.groups import *
from emlp_pytorch.reps import *
from emlp_pytorch.nn import EMLP, EMLPBlock, Linear
from emlp_pytorch.nn import uniform_rep

from .utils import index_to_perm
import time

class Generator(nn.Module): 
    def __init__(self, m = 200, ch = 96, num_layers = 3, noise_scale=1.0, dz = 20, threshold=0.1, custom_intermediate=False, device=None): 
        super(Generator, self).__init__()
        self.m = m
        self.dz = dz
        self.noise_scale = noise_scale
        self.threshold = threshold
        self.custom_intermediate = custom_intermediate
        self.device=device
        
        self.get_reps(ch, num_layers)
        self.model = nn.Sequential(
            *[EMLPBlock(rin, rout) for rin, rout in zip(self.reps, self.reps[1:])],
            Linear(self.reps[-1], self.rep_out)).to(self.device)
    
    def get_intermediate_reps(self, ch, G):
        """
        Algorithm: From V0 to V3, putting more weight on V0, V1 and V3
        """
        if ch == 32:
            # uniform_rep: 8V⁰+4V+2V²+V³
            rep = 10*T(0)(G) + 5*T(1)(G) + T(2)(G) + T(3)(G)
            return rep
        elif ch == 64:
            # uniform_rep: 16V⁰+8V+4V²+2V³
            rep = 20*T(0)(G) + 10*T(1)(G) + 2*T(2)(G) + 2*T(3)(G)
            return rep
        elif ch == 96:
            # uniform_rep: 22V⁰+11V+5V²+2V³+V⁴
            rep = 30*T(0)(G) + 15*T(1)(G) + 3*T(2)(G) + 3*T(3)(G)
            return rep
        elif ch == 128:
            # uniform_rep: 30V⁰+15V+7V²+3V³+V⁴
            rep = 40*T(0)(G) + 20*T(1)(G) + 4*T(2)(G) + 4*T(3)(G)
            return rep
        else:
            raise NotImplemented
        
    def get_reps(self, ch:int, num_layers: int):
        G = SO(2).to(self.device)
        self.G = G
        self.rep_in = self.m * T(0)(G) + (self.m + self.dz) * T(1)(G)
        if self.custom_intermediate:
            rep_mids = num_layers * [self.get_intermediate_reps(ch, G)]
        else: 
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
        # images: [B, 1, 28, 28]
        B = images.shape[0]
        images = images[:,0,:,:] # [B, 28, 28]
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
        
    def forward(self, images): 
        # images: [B, 1, 28, 28]
        B = images.shape[0]
        v = self.get_vec(images) # [B, m, 3]
        Z = self.sample_noise(B) # [B, 2, dz]
        
        coord_flat = v[:,:,:2] # [B, m, 2]
        coord_flat = coord_flat.flatten(start_dim=1) # [B, 2*m]
        Z_flat = Z.transpose(1, 2).flatten(start_dim=1) # [B, 2*dz]
        
        x = torch.zeros(B, 3*self.m + 2*self.dz).to(self.device) # [B, 3*m + 2*dz]
        x[:, :self.m] = v[:,:,2] 
        x[:, self.m : 3*self.m] = coord_flat 
        x[:, 3*self.m:] = Z_flat
        
        out = self.model(x) # [B, 4] 
        out = out.reshape(B, 2, 2).transpose(-1, -2) # [B, 2, 2]
        # checked
        return out
    