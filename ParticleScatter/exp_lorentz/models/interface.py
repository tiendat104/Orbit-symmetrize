import sys
import os
cwd = os.getcwd() + '/'
root_dir = os.path.join(cwd, os.pardir) + '/'
sys.path.append(os.path.dirname(root_dir + "emlp-pytorch/"))

import torch
import torch.nn as nn  # pylint: disable=R0402:consider-using-from-import
import torch.nn.functional as F
import numpy as np
import logging

from emlp_pytorch.groups import *
from emlp_pytorch.reps import T, Rep, Scalar, bilinear_weights, SumRep
from emlp_pytorch.nn import uniform_rep, EMLPBlock, Linear
from .backbone import MLP
from .generator import BasicEMLP, ScalarEMLP
from .utils import get_inverse_Lorentz, compute_loss_bound

class InvariantPoly(nn.Module):
    def __init__(self,):
        super(InvariantPoly,self).__init__()
    
    def forward(self, M): 
        # M: [B, 4, 4]
        # return [B, 16]
        B = M.shape[0]
        Q = -torch.eye(4, dtype=torch.float, device=M.device) # [4, 4]
        Q[0,0] = 1.0 
        Q = Q.unsqueeze(0).expand(B, 4, 4) # [B, 4, 4]
        
        C = torch.matmul(M.transpose(-1, -2), Q) # [B, 4, 4]
        C = torch.matmul(C, M) # [B, 4, 4]
        C = C.flatten(start_dim=1) # [B, 16]
        
        return C # [B, 16]

class LorentzInterface(nn.Module):
    def __init__(self, args=None, generator="scalar", ch=96, num_layers=3, noise_scale=0.1, dz=1, device=None):
        super(LorentzInterface, self).__init__()
        self.device = device 
        self.G = Lorentz().to(device)
        self.dz = dz
        self.noise_scale = noise_scale

        if generator == "basic":
            self.generator = BasicEMLP(ch, num_layers, dz)
        elif generator == "scalar":
            self.generator = ScalarEMLP(ch, num_layers, dz, activation=args.gen_act, batchnorm=args.gen_bn, last_act=args.last_act, gen_type=args.gen_type)
        self.generator.to(device)
        
        self.invariant_poly = InvariantPoly().to(self.device)
        if args.loss_type == 1:
            self.interface_criterion = torch.nn.L1Loss(reduction="mean").to(self.device)
        elif args.loss_type==2:
            self.interface_criterion =torch.nn.MSELoss(reduction="mean").to(self.device)
        
    def sample_noise(self, B):
        # Compactly distributed noise 
        Z = self.noise_scale * torch.zeros(B, self.dz, device=self.device).uniform_(-1.0, 1.0)
        return Z
    
    def formatize_input(self, x, Z): 
        # x: [B, 16], Z: [B, dz]
        B = x.shape[0] 
        
        v = torch.zeros(B, self.dz+16, device=self.device) # [B, dz+16]
        v[:,:self.dz] = Z 
        v[:,self.dz:] = x 
        return v # [B, dz+16]
    
    def get_orbit_distance(self, gs):
        # gs: [B, 4, 4]
        B = gs.shape[0]
        # orbit distance 
        gen_score = self.invariant_poly(gs) # [B, 16]
        
        valid_orbit = torch.eye(4, device=self.device)[None,:,:] # [1, 4, 4]
        valid_score = self.invariant_poly(valid_orbit) # [1, 16]
        valid_score = valid_score.expand(B, -1) # [B, 16]
        
        orbit_distance = self.interface_criterion(gen_score, valid_score) # scalar
        
        # checked
        return orbit_distance
    
    def forward(self, x):
        # x: [B, 16]
        B = x.shape[0]
        Z = self.sample_noise(B) # [B, dz]
        v = self.formatize_input(x, Z) # [B, dz+16]
        
        gs = self.generator(v) # [B, 4, 4]
        
        # get orbit distance 
        orbit_distance = self.get_orbit_distance(gs) # scalar
        # checked

        return gs, orbit_distance

class EquiLorentzNet(nn.Module):
    def __init__(self, args=None, device=None):
        super().__init__()
        self.rep_in = 4*T(1) 
        self.rep_out = T(0)
        self.G = Lorentz().to(device)
        
        self.backbone = MLP(self.rep_in, self.rep_out, self.G, ch=args.mlp_ch, num_layers=args.mlp_L, device=device) 
        self.interface = LorentzInterface(args=args, generator=args.generator, ch=args.gen_ch, num_layers=args.gen_L, noise_scale=args.gen_Zscale, dz=args.gen_dz, device=device)
        self.sample_size = args.sample_size
        self.bound = args.bound
        
    def transform_input(self, x, gs): # checked
        # x: [B, 16]
        # gs: [B, 4, 4]
        B = x.shape[0]
        gs_inv = get_inverse_Lorentz(gs)  # [B, 4, 4]
        
        x = x.reshape(B, 4, 4).transpose(-1, -2) # [B, 4, 4]
        transformed_x = torch.matmul(gs_inv, x) # [B, 4, 4]
        transformed_x = transformed_x.transpose(-1, -2).flatten(start_dim=1) # [B, 16]
        
        return transformed_x # [B, 16]
    
    def forward(self, x): # checked
        # x: [B, 16]
        B = x.shape[0]

        # Prepare for interface
        x = x.unsqueeze(1).expand(B, self.sample_size, 16) # [B, k, 16]
        x = x.reshape(B*self.sample_size, 16) # [B*k, 16]
        
        # sample from p(g|x)
        gs, orbit_distance = self.interface(x) 
        # gs: [B*k, 4, 4]
        loss_bound = compute_loss_bound(gs, self.bound)
        
        # transform input
        transformed_x = self.transform_input(x, gs) # [B*k, 16] 
        logits = self.backbone(transformed_x) # [B*k, 1] 
        logits = logits.reshape(B, self.sample_size) # [B, k]
        logits = logits.mean(dim=1) # [B,]
        logits = logits.unsqueeze(-1) # [B, 1]
        
        return logits, orbit_distance, loss_bound