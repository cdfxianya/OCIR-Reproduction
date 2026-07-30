"""
Code source: https://github.com/karpathy/pytorch-made
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from src.blocks.more_flows import made
from src.blocks import src_utils
class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """
    
    def __init__(self, dim, parity, net_class=made.ARMLP, nh=24):
        super().__init__()
        self.dim = dim
        self.net = net_class(dim, dim*2, nh) # (in dim, out dim, hidden dim)
        self.parity = parity
    def forward(self, x, ldj):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        z = x * torch.exp(s) + t
        # reverse order, so if we stack MAFs correct things happen
        z = z.flip(dims=(1,)) if self.parity else z
        ldj += torch.sum(s, dim=1)
        return z, ldj
    
    def backward(self, z, ldj):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z)
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone()) # clone to avoid in-place op errors if using IAF
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            ldj -= s[:, i]
        return x, ldj

class IAF(MAF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead, 
        where sampling will be fast but density estimation slow
        """
        self.forward, self.backward = self.backward, self.forward