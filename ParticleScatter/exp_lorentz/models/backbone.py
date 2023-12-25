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

class Linear_jax_init(nn.Linear):
    """ Linear layer for equivariant representations. """

    def __init__(self, cin, cout):
        super().__init__(cin, cout)
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

def MLPBlock(cin, cout):
    """ Basic building block of MLP consisting of Linear, SiLU (Swish), and dropout. """
    return nn.Sequential(Linear_jax_init(cin, cout), nn.SiLU())

class MLP(nn.Module):
    """ Standard baseline MLP. Representations and group are used for shapes only. """

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3, device='cuda'):
        super().__init__()
        group = group.to(device)
        self.G = group
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = nn.Sequential(
            *[MLPBlock(cin, cout) for cin, cout in zip(chs, chs[1:])],
            Linear_jax_init(chs[-1], cout)
        )
        self.net.to(device)

    def forward(self, x):
        """ Forward pass of MLP. """
        y = self.net(x)
        return y
