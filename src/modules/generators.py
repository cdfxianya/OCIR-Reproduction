'''
Decoder, f_D

Generator, G

Note that the decoder and generator in OCIR shares the same parameters
'''

import torch.nn.functional as F
import torch.nn as nn

import src.blocks as bs
from src.modules import distributions

class Decoder(nn.Module):
    def __init__(self, dx:int, dz:int, dc:int, window:int, 
                 d_model:int, num_heads:int, p_h, p_c = None,
                 p_c2 = None, dc2: int = 0
                ): 
        super(Decoder, self).__init__() 
        self.depth = 2
        self.window = window
        self.dc = dc #if p_c is not None else 0
        self.dz = dz
        self.dc2 = dc2
        self.num_heads = num_heads

        self.latent_decoder = bs.wide_decoding(dz, self.dc, hidden_dim= d_model,
                                               window = window, dc2 = self.dc2)
        TransformerDecoder = [bs.TransformerEncoderBlock(embed_dim = d_model, num_heads = self.num_heads,
                                                                       ff_hidden_dim = int(d_model * 3), dropout = 0.15,
                                                                       prenorm = True) for _ in range(self.depth)]
        self.TransformerDecoder = nn.ModuleList(TransformerDecoder)

        self.final_layer = bs.Linear(d_model, dx, bias=False)
        
        # self.layernorm = nn.LayerNorm(dz)
        # p_h(z) & p(c)
        self.h = p_h
        self.p_c = p_c
        self.p_c2 = p_c2
    def forward(self, z, c = None, generation = False, zin = None, c2 = None):
        # z (N, dz)
        # c (N, L, dc)
        if not generation: 
            # At Inference where the arg c is logit inferred by f_C
            if c is not None:
                if isinstance(self.p_c, distributions.DiscreteUniform):
                    c = F.softmax(c,dim = -1)#.detach()
                elif isinstance(self.p_c, distributions.UniformDistribution):
                    c = c#.detach()
                else:
                    c = c#.detach()
                    
                if c2 is not None:
                    if isinstance(self.p_c2, distributions.DiscreteUniform):
                        c2 = F.softmax(c2,dim = -1).detach()
                    else:
                        c2 = c2.detach()
        z_tokens = self.latent_decoder(z, c, zin, c2 = c2)
        
        for layer in self.TransformerDecoder:
            z_tokens = layer(z_tokens)
        x = self.final_layer(z_tokens)
        return x 
    
    def generation(self, num_samples, fixed_code = None):
        # N is the target shape in tuple
        
        N_c = (num_samples, self.window, self.dc)
        N_z = (num_samples, self.dz)
        
        if self.p_c2 is not None:
            N_c2 = (num_samples, self.dc2)
        # sample z ~ p(z') -> p_h(z)
        if isinstance(self.h, distributions.DiagonalGaussian):
            z = self.h.sample(N_z) # (N, L, dz)
            z0 = None
        else:
            z, z0, logdet = self.h.sample(N_z) # (N, L, dz)
        
        # sample c ~ p(c)
        if self.p_c is not None:
            c = self.p_c.sample(N_c, fixed_code) # (N, L, dc)
            if self.p_c.gumbel:
                c, c_logit = c
            else:
                c_logit = None
        else:
            c, c_logit = None, None     
        
        if self.p_c2 is None:
            x = self.forward(z, c, generation = True)
            return x, [z,z0, c, c_logit], logdet
        else:
            c2 = self.p_c2.sample(N_c2, fixed_code) # (N, dc2)
            if self.p_c.gumbel:
                c2, c2_logit = c2
            else:
                c2_logit = None    
            
            x = self.forward(z, c, generation = True,c2 = c2)
            return x, [z,z0, c, c_logit, c2, c2_logit], logdet
    
        