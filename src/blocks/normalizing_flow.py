'''
(RNVP)  Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803


(MAF) Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017 
https://arxiv.org/abs/1705.07057


(IAF)
Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934

(Linear flow)
Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import linalg
from src.blocks.more_flows import maf
from src.blocks import src_utils
def create_checkerboard_mask(h, seq = False, invert=False):
    '''
    sequence wise mask if a sequence of (N,L,C) is dealt with other wise it is also 
    a channel wise with checkrboard pattern. 
    '''
    x, y = torch.arange(h, dtype=torch.int32), torch.arange(1, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    mask = torch.fmod(xx + yy, 2)
    if seq:
        mask = mask.to(torch.float32).view(1, h, 1)
    elif not seq:
        mask = mask.to(torch.float32).view(1, h)
    if invert:
        mask = 1 - mask
    return mask

def create_channel_mask(c_in, seq = False, invert=False):
    # Channel wise mask 
    mask = torch.cat([torch.ones(c_in//2, dtype=torch.float32),
                      torch.zeros(c_in-c_in//2, dtype=torch.float32)])
    if seq:
        mask = mask.view(1, 1, c_in)
    elif not seq:
        mask = mask.view(1, c_in)
    if invert:
        mask = 1 - mask
    return mask

class st_block(nn.Module):
    '''
    A simple neural network for parameterizing s and t
    
    The final output of the network needs to be (dz * 2) as the s and t get split by 2 in the flow layer.  
    '''
    def __init__(self, dz = 10, hidden_dim = 64):
        super(st_block, self).__init__() 
        self.param_s_t = nn.Sequential(src_utils.Linear(dz, hidden_dim, zero= False ),
                                       nn.LeakyReLU(0.2),
                                       src_utils.Linear(hidden_dim, hidden_dim, zero= False ),
                                       nn.LeakyReLU(0.2),
                                       src_utils.Linear(hidden_dim, dz * 2 , zero= True))

    def forward(self, z):
        return self.param_s_t(z)
class Linear_flow(nn.Module):
    def __init__(self, z_dim = 10):
        super(Linear_flow, self).__init__() 
        '''
        TODO: reference
        PL-linear flow z_tar = W*z where W = PLU
        The implementation is based on LF used in "Glow: Generative Flow with Invertible 1x1 Convolutions"        
        '''
        W_init = torch.Tensor(z_dim, z_dim) # Random orthogonal matrix
        nn.init.orthogonal_(W_init) 
        
        P,L,U = linalg.lu(W_init.numpy())
        
        s = np.diag(U)
        sign_s = np.sign(s)
        log_s = np.log(abs(s))
        U = np.triu(U, k=1)
        self.z_dim = z_dim
        self.mask_U = torch.triu(torch.ones_like(W_init), 1)
        self.mask_L = torch.tril(torch.ones_like(W_init), -1)
        self.register_buffer("P", torch.from_numpy(P))
        self.register_buffer("I", torch.eye(z_dim))
        self.register_buffer("sign_s", torch.from_numpy(sign_s))

        self.LF_L = nn.Parameter(torch.from_numpy(L))
        self.LF_U = nn.Parameter(torch.from_numpy(U))
        self.LF_S = nn.Parameter(torch.from_numpy(log_s))
    def forward(self, z, ldj):
        '''Notice that for consistency, the forward computation of our LF is from z -> z_tar which is 
        opposite to those specified in Glow, although this essentially makes no differece 
        (only difference is in the notion of what is taken as input for forward and backward paths)
        '''
        L_ = self.LF_L * self.mask_L.to(z.device) + self.I
        U_ = self.LF_U * self.mask_U.to(z.device) + torch.diag(self.sign_s * torch.exp(self.LF_S))
        W = self.P @ L_ @ U_ 
        ldj += self.logdet_()
        return z @ W.t(), ldj
    def inverse(self, z_tar, ldj):
        L_ = self.LF_L * self.mask_L.to(z_tar.device) + self.I
        U_ = self.LF_U * self.mask_U.to(z_tar.device) + torch.diag(self.sign_s * torch.exp(self.LF_S))
        # 1.
        W = self.P @ L_ @ U_ 
        W = W.inverse()
        # 2. 
        # W = U_.inverse() @ L_.inverse() @ self.P.inverse()
        ldj -= self.logdet_()
        return z_tar @ W.t(), ldj # the output dimension can be of either (N,T,d_z) for parallel process or (N,d_z) for each time step i
    def logdet_(self):
        # The way to compute logdet is the same in both forward and inverse paths
        # We need to compute logdet only once regardless of whether the input comes recursively.
        return torch.sum(self.LF_S) # Need to consider the dimension of actual input to flow
    def __repr__(self):
        return f'{self.__class__.__name__}({self.z_dim}, {self.z_dim})'        

class CouplingLayer(nn.Module):

    def __init__(self, network, mask, c_in):
        """
        Coupling layer inside a normalizing flow.
        """
        super().__init__()
        self.network = network
        
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        # self.scale = torch.nn.utils.parametrizations.weight_norm(nn.Linear(c_in, c_in, bias = True))
        
        # Register mask as buffer as it is a tensor which is not a parameter,
        # but should be part of the modules state.
        self.register_buffer('mask', mask)

    def forward(self, z, ldj):
        # z is shape of (N, dz)
        # Apply network to masked input
        z_in = z * self.mask
        nn_out = self.network(z_in)
        s, t = nn_out.chunk(2, dim=-1)
        
        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1) # (1, c)
        s = torch.tanh(s * s_fac) / s_fac
        # s = self.scale(torch.tanh(s))
        
        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        z = (z + t) * torch.exp(s) 
        ldj += s.sum(dim=[1])
        return z, ldj
    
    def inverse(self, z, ldj):

        z_in = z * self.mask
        nn_out = self.network(z_in)
        s, t = nn_out.chunk(2, dim=-1)
        
        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1) # (1, c)
        s = torch.tanh(s * s_fac) / s_fac
        # s = self.scale(torch.tanh(s))
        
        # s = self.scaling_factor(F.tanh(s))
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        
        z = (z  * torch.exp(-s) ) - t
        ldj -= s.sum(dim=[1]) # (N, )
        return z, ldj
    
class NormalizingFlow(nn.Module):
    def __init__(self, 
                 transform = "MAF",
                 dz = 10,
                 num_layers = 6,
                 linear_flow = False): # transforms is a list of flow layers
        super().__init__()
        hidden_dim = 128
        flows = []
        for i in range (num_layers):
            if linear_flow:
                flows.append(Linear_flow(z_dim = dz))
                
            if transform == "RNVP":
                flows.append(CouplingLayer(network=  st_block(dz, hidden_dim= hidden_dim), 
                                mask = create_checkerboard_mask(h = dz, seq = False, invert=(i%2==1)),
                                c_in = dz))
            elif transform == "MAF":
                flows.append(maf.MAF(dim = dz, parity = i%2==1, nh = hidden_dim)) 
            
            elif transform == "IAF":
                flows.append(maf.IAF(dim = dz, parity = i%2==1, nh = hidden_dim))
            
        self.flows = nn.ModuleList(flows)
    def forward(self, z0):  # z0 -> zK
        ldj = torch.zeros(z0.shape[0], device = z0.device)
        z = z0
        for transform in self.flows:
            z, ldj = transform(z, ldj) #transform returns updated z and log det jacobian
        return z, ldj

    def inverse(self, z): # zK -> z0 
        ldj = torch.zeros(z.shape[0], device = z.device)
        for transform in reversed(self.flows):
            z, ldj = transform.inverse(z, ldj)
        z0 = z  
        return z0, ldj
    
    
if __name__ == '__main__':
    N, dz = 3, 16 
    xx = torch.randn((N, dz)) * 0.02

    m = NormalizingFlow("RNVP", dz = dz)
    print(m)
    
    yy, ldj = m(xx)
    
    zz, ldj_inv = m.inverse(yy)
    
    if torch.allclose(xx, zz, rtol = 1e-4):
        print("invertible")
        print(xx)
        print(yy)