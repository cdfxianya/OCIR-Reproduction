'''
Encoder f_E

Encoder f_C

'''
import torch.nn.functional as F
import torch
import torch.nn as nn

from src.blocks import transformers, src_utils, tcns
from src.modules import distributions
class LatentEncoder(nn.Module):
    def __init__(self, dx:int, dz:int, window:int, 
                 d_model:int, num_heads:int, z_projection:str, time_emb:bool,
                 encoder_E:str, p_h, shared_EC = False
                 ): 
        super(LatentEncoder, self).__init__() 
        
        self.z_projection = z_projection
        self.num_heads = num_heads
        self.depth = 2
        self.time_emb = time_emb
        self.shared_EC = shared_EC
        self.d_model = d_model
        if encoder_E == "transformer":
            if not self.shared_EC:
                # Suppose the longest length of cm data could be 520
                if self.time_emb:
                    self.time_embedding = nn.Embedding(num_embeddings= 550, embedding_dim = d_model)
                self.pos_enc = transformers.SinCosPositionalEncoding(d_model, window + 2)
                self.fE_projection = src_utils.Linear(dx, d_model)

                TransformerEncoder = [transformers.TransformerEncoderBlock(embed_dim = d_model, num_heads = self.num_heads,
                                                                ff_hidden_dim = int(d_model * 3), dropout = 0.1,
                                                                prenorm =True) for _ in range(self.depth)]
                self.TransformerEncoder = nn.ModuleList(TransformerEncoder)
            else:
                TransformerEncoder = [transformers.TransformerEncoderBlock(embed_dim = d_model, num_heads = self.num_heads,
                                                                ff_hidden_dim = int(d_model * 3), dropout = 0.1,
                                                                prenorm =True) for _ in range(self.depth)]
                self.TransformerEncoder = nn.ModuleList(TransformerEncoder) #self.make_MLP()
            # (N, L, C) -> (N, C)
            if z_projection == "aggregation":
                self.latent_aggregation = transformers.Aggregation(d_model, method = 'weighted')
                
            elif (z_projection == "spc") and (not self.shared_EC):
                # otherwise, a BERT-style special token (namely compressive token) is done 
                self.compressive_token = nn.Parameter(torch.randn(1,1,d_model))
                
            elif z_projection == "rnn":
                self.latent_aggregation = src_utils.rnn_aggregation(d_model, d_model)
            elif z_projection == "seq" and (not self.shared_EC):
                self.compressive_token = nn.Parameter(torch.randn(1,1,d_model))
                self.projection_zin = src_utils.Linear(d_model, dz )
            elif z_projection == "aggregation_all":
                self.latent_aggregation = transformers.Aggregation_all(d_model, window + 1 if self.time_emb else window)
                
        if p_h is None:
            self.mu = src_utils.Linear(d_model, dz , bias=True) # torch.nn.utils.parametrizations.weight_norm(src_utils.Linear(d_model, dz , bias=True))
        else:
            self.mu = torch.nn.utils.parametrizations.weight_norm(src_utils.Linear(d_model, dz , bias=True)) # src_utils.Linear(d_model, dz , bias=True) # torch.nn.utils.parametrizations.weight_norm(src_utils.Linear(d_model, dz , bias=True))
            self.logvar = src_utils.Linear(d_model, dz , bias=True) # torch.nn.utils.parametrizations.weight_norm(src_utils.Linear(d_model, dz , bias=True))

        self.p_h = p_h
    def forward(self, x, tidx = None):
        N, L, _ = x.shape
        if not self.shared_EC:
            x_proj = self.fE_projection(x)
            
            # Time index embeddings
            if self.time_emb:
                if tidx is not None: 
                    time_token = self.time_embedding(tidx) # (N, 1, d_model)
                else:
                    # For generator-to-encoder inference the time index is not known, so time index 0 is assigned as a unknown time. 
                    time_token = self.time_embedding(torch.zeros((N,1), dtype=torch.long).to(x.device)) # zero token
                x_emb = torch.cat((x_proj,time_token), dim = 1)
            else: x_emb = x_proj
            # add apc as a compressive token
            if self.z_projection == "spc" or (self.z_projection == "seq"):
                compressive_token = self.compressive_token.expand(N,-1,-1) # TODO repeat?
                # compressive_token = self.compressive_token.repeat(N,1,1) # TODO repeat?

                x_emb = torch.cat((compressive_token, x_emb), dim = 1)  
                        
            # positional encoding
            x_emb = self.pos_enc(x_emb)
            
            # transformer
            for layer in self.TransformerEncoder:
                x_emb = layer(x_emb)
                
        else:
            x_emb = x
            for layer in self.TransformerEncoder:
                x_emb = layer(x_emb)

        if self.z_projection == "spc":
            z = x_emb[:,0,:] # (N, d_modle)
            z_in = None
        elif self.z_projection == "seq":
            pass
            z = x_emb[:,0,:] # (N, d_modle)
            z_in = x_emb[:,1:,:] if not self.time_emb else x_emb[:,1:-1,:] # (N, L, d_modle)
            assert z_in.shape[1] == L
            z_in = self.projection_zin(z_in)
        else:
            z = self.latent_aggregation(x_emb) # (N, d_modle)
            z_in = None

        if self.p_h is None:
            mu = self.mu(z)
            log_var = None
        else:
            mu = self.mu(z)
            log_var = self.logvar(z) 

        return mu, log_var, z_in
    
    def reparameterization_NF(self, mu, log_var):
        # Stochastic
        # reparameterization trick with flow transform
        if isinstance(self.p_h, distributions.DiagonalGaussian):
            z = self.p_h.sample(log_var, mu, log_var)
            logdet, z0 = None, None
        else:
            z, logdet, z0 = self.p_h(log_var, mu, log_var)
        return z, logdet, z0
    
    # TODO get rid of zin thingy everywhere
    def encoding(self, x, tidx, zin = False):
        # Deterministic inference
        
        mu, logvar, z_in = self.forward(x, tidx)
        if not zin: 
            if isinstance(self.p_h, distributions.DiagonalGaussian):
                return mu
            else:
                z, _, z0 = self.p_h(z0 = mu)
                return z
        else:
            if isinstance(self.p_h, distributions.DiagonalGaussian):
                return mu, z_in
            else:
                z, _, z0 = self.p_h(z0 = mu)
                return z, z_in
            
    def timeembedding(self, tidx):
        if self.time_emb:
            time_token = self.time_embedding(tidx)
            return time_token 
        else:
            return None
        
    def make_MLP(self):
        mlp = nn.Sequential(src_utils.Linear(self.d_model, self.d_model),
                            nn.LeakyReLU(0.2),
                            src_utils.Linear(self.d_model, self.d_model)
                            )
        return mlp 
class CodeEncoder(nn.Module):
    def __init__(self, dx:int, dc:int,
                 d_model:int, c_type:str, c_posterior_param:str,
                 shared_EC = False,
                 c_kl = False
                 ): 
        super(CodeEncoder, self).__init__() 
        self.dx = dx
        self.d_model_c = d_model
        self.depth = 2
        self.dc = dc
        self.c_type = c_type
        self.shared_EC = shared_EC
        # Fix this to "hard fitting" for one-to-one cycle-consistency 
        self.c_posterior_param = c_posterior_param
        
        if not self.shared_EC:
            code_encoder = [self.make_MLP() for _ in range(self.depth)]
            self.code_encoder = nn.ModuleList(code_encoder)
            self.fC_projection = src_utils.Linear(dx, d_model) #transformers.Conv1by1(dx, d_model)
        else:
            self.code_encoder = self.make_MLP()
            
        if c_type == "discrete":
            self.classifier = src_utils.Linear(self.d_model_c,self.dc, bias=False)
            self.softmax = nn.Softmax(-1)
        elif c_type == "continuous":
            # soft fitting
            if self.c_posterior_param == "soft":
                self.code_mu = src_utils.Linear(d_model, self.dc, bias=False) # torch.nn.utils.parametrizations.weight_norm(src_utils.Linear(self.d_model_c, self.dc * 2, bias=False))
                self.code_logvar = src_utils.Linear(d_model, self.dc, bias=False)
            # hard fitting
            else: 
                self.code_mu =  src_utils.Linear(self.d_model_c,self.dc, bias=False) # torch.nn.utils.parametrizations.weight_norm(src_utils.Linear(self.d_model_c,self.dc, bias=False))

    def forward(self, x):
        N, L, c = x.shape
        if not self.shared_EC:
            code_emb = self.fC_projection(x)
            
            for layer in self.code_encoder:
                code_emb = layer(code_emb)
        else:
            code_emb = self.code_encoder(x)
            
            
        if self.c_type == "discrete":
            logit = self.classifier(code_emb)
            return logit.view(N,L,self.dc), None
        elif self.c_type == "continuous":
            if self.c_posterior_param == "soft":
                mu = self.code_mu(code_emb)
                log_var = self.code_logvar(code_emb)
                return mu.view(N,L,self.dc), log_var.view(N,L,self.dc)
            
            else:
                mu = self.code_mu(code_emb)
                return mu.view(N,L,self.dc), None
        
    def inference(self, x, logits = False):
        c, c_log_var = self.forward(x)
        if self.c_type == "discrete":
            return torch.argmax(self.softmax(c),dim = -1, keepdim=True) if not logits else c #  
        else:
            return c
    def reparameterization(self, mu, log_var):
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        c = mu + eps * std  # Gaussian sample

        # Apply truncation to approximate uniform distribution
        c = torch.clamp(c, min=-1, max=1)
        return c
    def make_MLP(self):
        mlp = nn.Sequential(src_utils.Linear(self.d_model_c, self.d_model_c),
                            nn.LeakyReLU(0.2),
                            src_utils.Linear(self.d_model_c, self.d_model_c)
                            )
        return mlp 
    
class CodeEncoder_seq(nn.Module):
    def __init__(self, dx:int, dc:int,
                 d_model:int, c_type:str, c_posterior_param:str,
                 shared_EC = False, c2_projection = "aggregation_all", window = 25,
                 time_emb = False
                 ): 
        super(CodeEncoder_seq, self).__init__() 
        self.dx = dx
        self.d_model_c = d_model
        self.depth = 2
        self.dc = dc
        self.c_type = c_type
        self.shared_EC = shared_EC
        self.c2_projection = c2_projection
        self.time_emb = time_emb
        # Fix this to "hard fitting" for one-to-one cycle-consistency 
        self.c_posterior_param = c_posterior_param
        
        if not self.shared_EC:
            # Suppose the longest length of cm data could be 520
            if self.time_emb:
                self.time_embedding = nn.Embedding(num_embeddings= 550, embedding_dim = d_model)
                # src_utils.init_embedding(self.time_embedding)
            self.pos_enc = transformers.SinCosPositionalEncoding(d_model, window + 2)
            self.fC2_projection = src_utils.Linear(dx, d_model) #transformers.Conv1by1(dx, d_model)

            TransformerEncoder = [transformers.TransformerEncoderBlock(embed_dim = d_model, num_heads = 4,
                                                            ff_hidden_dim = int(d_model * 3), dropout = 0.25,
                                                            prenorm =True) for _ in range(self.depth)]
            self.TransformerCodeEncoder = nn.ModuleList(TransformerEncoder)
        else:
            self.TransformerCodeEncoder = self.make_MLP()
        
        if c2_projection == "aggregation_all":
            self.code2_aggregation = transformers.Aggregation_all(d_model, window + 1 if self.time_emb else window)
                
        elif (c2_projection == "spc") and (not self.shared_EC):
            # otherwise, a BERT-style special token (namely compressive token) is done 
            self.fault_token = nn.Parameter(torch.randn(1,1,d_model))
                
        if c_type == "discrete":
            self.classifier = src_utils.Linear(self.d_model_c,self.dc, bias=False)
            self.softmax = nn.Softmax(-1)

    def forward(self, x, tidx = None):
        N, L, c = x.shape
        if not self.shared_EC:
            x_proj = self.fC2_projection(x)
            
            # Time index embeddings
            if self.time_emb:
                if tidx is not None: 
                    time_token = self.time_embedding(tidx) # (N, 1, d_model)
                    # print(time_token.device)
                else:
                    # For generator-to-encoder inference the time index is not known, so time index 0 is assigned as a unknown time. 
                    time_token = self.time_embedding(torch.zeros((N,1), dtype=torch.long).to(x.device)) # zero token
                x_emb = torch.cat((x_proj,time_token), dim = 1)
            else: x_emb = x_proj
            # add apc as a compressive token
            if self.c2_projection == "spc":
                fault_token = self.fault_token.expand(N,-1,-1) 
                x_emb = torch.cat((x_emb, fault_token), dim = 1)  
                        
            # positional encoding
            x_emb = self.pos_enc(x_emb)
            
            # transformer
            for layer in self.TransformerCodeEncoder:
                x_emb = layer(x_emb)
                
        else:
            x_emb = x
            x_emb = self.TransformerCodeEncoder(x_emb)
        if self.c2_projection == "spc":
            code_emb = x_emb[:,-1,:] # (N, d_modle)
        else:
            
            code_emb = self.code2_aggregation(x_emb) # (N, d_modle)
        
        if self.c_type == "discrete":
            logit = self.classifier(code_emb)
            return logit
        
    def inference(self, x, logits = False):
        c = self.forward(x)
        if self.c_type == "discrete":
            return torch.argmax(self.softmax(c),dim = -1, keepdim=True) if not logits else c #  
        else:
            return c
    def make_MLP(self):
        mlp = nn.Sequential(src_utils.Linear(self.d_model_c, self.d_model_c),
                            nn.LeakyReLU(0.2),
                            src_utils.Linear(self.d_model_c, self.d_model_c))
        return mlp 