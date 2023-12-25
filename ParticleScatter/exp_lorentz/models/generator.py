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
from .utils import get_inverse_Lorentz
from .modules import compute_generating

def get_basic_EMLP(ch=96, num_layers=3, dz=10):
    G = Lorentz()
    rep_in = dz * T(0)(G) + 4*T(1)(G)
    rep_out = 4*T(1)(G)
    rep_mids = num_layers * [uniform_rep(ch, G)]
    reps = [rep_in] + rep_mids
    
    net = nn.Sequential(
            *[EMLPBlock(rin, rout) for rin, rout in zip(reps, reps[1:])],
            Linear(reps[-1], rep_out))
    return net

class BasicEMLP(nn.Module):
    def __init__(self, ch=96, num_layers=3, dz=10):
        super(BasicEMLP, self).__init__()
        self.G = Lorentz()
        self.dz = dz
        self.get_reps(ch, num_layers)
        self.net = nn.Sequential(
            *[EMLPBlock(rin, rout) for rin, rout in zip(self.reps, self.reps[1:])],
            Linear(self.reps[-1], self.rep_out))
        
    def get_reps(self, ch: int, num_layers: int):
        self.rep_in = self.dz * T(0)(self.G) + 4*T(1)(self.G)
        self.rep_out = 4*T(1)(self.G)
        rep_mids = num_layers * [uniform_rep(ch, self.G)]
        self.reps = [self.rep_in] + rep_mids 
    
    def forward(self, v):
        # v: [B, dz+16]
        B = v.shape[0]
        v = self.net(v) # [B, 16]
        v = v.reshape(B, 4, 4).transpose(-1, -2) # [B, 4, 4]
        return v # [B, 4, 4]

class Linear_init(nn.Linear):
    def __init__(self, cin, cout, activation):
        super().__init__(cin, cout)
        if activation == "leaky_relu":
            nn.init.kaiming_uniform_(self.weight, nonlinearity="leaky_relu")
        else:
            nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.zeros_(self.bias) 
        
def get_activation(activation="relu"):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "Prelu":
        return nn.PReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "elu":
        return nn.ELU()
    
def MLPBlock(cin: int, cout: int, activation="relu", batchnorm=False):
    fc = Linear_init(cin, cout, activation) 
    bn = nn.BatchNorm1d(cout, track_running_stats=False)
    act = get_activation(activation)
    if batchnorm:
        layer = nn.Sequential(fc, bn, act)
    else:
        layer = nn.Sequential(fc, act)
    return layer

class MLP(nn.Module):
    def __init__(self, gen_type=2, ch=96, num_layers=3, activation="relu", batchnorm=False):
        super(MLP, self).__init__()
        if gen_type == 1:
            chs = [16] + num_layers*[ch]
        elif gen_type == 2:
            chs = [65] + num_layers*[ch]
        elif gen_type == 4:
            chs = [1605] + num_layers*[ch]
        elif gen_type == 5:
            chs = [73] + num_layers*[ch]
            
        cout = 16
        self.net = nn.Sequential(
            *[MLPBlock(cin, cout, activation, batchnorm) for cin, cout in zip(chs, chs[1:])],
            Linear_init(chs[-1], cout, activation)
        )
    
    def forward(self, x): 
        # x: [B, 16]
        B = x.shape[0]
        x = self.net(x) # [B, 16]
        return x # [B, 16]

class ScalarEMLP(nn.Module):
    def __init__(self, ch=96, num_layers=3, dz=1, activation="relu", batchnorm=False, last_act="identity", gen_type=2):
        super(ScalarEMLP, self).__init__()
        self.G = Lorentz()
        self.rep_in = dz * T(0)(self.G) + 4*T(1)(self.G)
        self.rep_mid = 4*T(1)(self.G) 
        self.gen_type = gen_type
        
        self.emlp_layer = Linear(self.rep_in, self.rep_mid) 
        #self.emlp_layer = EMLPBlock(self.rep_in, self.rep_mid)
        self.mlp_layer = MLP(gen_type, ch, num_layers, activation, batchnorm) 
        self.get_last_act(last_act)
    
    def get_last_act(self, activation):
        if activation == "identity":
            self.last_act = nn.Identity()
        elif activation == "sigmoid":
            self.last_act = nn.Sigmoid()
        elif activation == "relu":
            self.last_act = nn.ReLU()
        elif activation == "silu":
            self.last_act = nn.SiLU()
        elif activation == "elu":
            self.last_act = nn.ELU()
        elif activation == "tanh":
            self.last_act = nn.Tanh()
    
    def forward(self, x):
        # x: [B, dz+16]
        # equivariant
        B = x.shape[0]
        M = self.emlp_layer(x) # [B, 16]
        M = M.reshape(B, 4, 4).transpose(-1, -2) # [B, 4, 4]
        
        # invariant
        C = compute_generating(M, self.gen_type) # [B, 16] or [B, 65]
        C = self.mlp_layer(C) # [B, 16]
        C = C.reshape(B, 4, 4) # [B, 4, 4]
        C = C + self.last_act(C) # [B, 4, 4]
        
        A = torch.matmul(C, M.transpose(-1, -2)) # [B, 4, 4]
        A = A.transpose(-1, -2) # [B, 4, 4]
        return A
        
        