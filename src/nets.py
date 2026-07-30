import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np
import src.modules as md # (encoders, generators, discriminator_Q, distributions, flow_transforms) 
from src.blocks import transformers


class InfoGAN(nn.Module):
    def __init__(self, 
                dx:int = 14, dz:int = 10, dc:int = 6, window:int = 25, d_model:int = 128, 
                num_heads:int = 4, z_projection:str = "aggregation", D_projection:str = "aggregation", 
                time_emb:bool = True, c_type:str = "discrete", c_posterior_param:str = "soft", encoder_E:str = "transformer",
                device = "cpu"): 
        super(InfoGAN, self).__init__()
        self.device = device
        self.c_type = c_type
        self.code_posterior_param = c_posterior_param
        # Prior Distributions p(z') and p(c)
        self.prior_z = md.DiagonalGaussian(dz, mean = 0, var = 1, device=device)
        self.prior_c = md.UniformDistribution(device=device) if c_type == "continuous" \
                else md.DiscreteUniform(dc, onehot = True, device=device) 
        self.h = md.LatentFlow(dz, self.prior_z)
        
        self.G = md.Decoder(dx=dx, dz=dz, dc=dc, window=window, d_model=d_model,
                                    num_heads=num_heads, p_h = self.h, p_c = self.prior_c)
        self.shared_net = transformers.shared_transformer(dx=dx, d_model=d_model, window=window, 
                                                    num_heads=num_heads, D_projection=D_projection)
        self.D = md.Discriminator(dx=dx, window=window, d_model=d_model,
                                               num_heads=num_heads, D_projection=D_projection,
                                               shared_layer= self.shared_net)
        self.Q = md.CodePosterior(dx=dx, dc=dc, d_model=d_model, c_type=c_type, 
                                               c_posterior_param=c_posterior_param,
                                               shared_layer=self.shared_net)
        
        
    def Loss_Discriminator(self, x):
        sample_size = x.shape[0]
        x_gen, _,_ = self.G.generation(sample_size)
        fake = self.D(x_gen)
        real = self.D(x)

        if real.dim() == 2:
            real_loss = torch.mean((real -1)**2)
            fake_loss = torch.mean((fake)**2) 
        elif real.dim() == 3:
            real_loss = torch.sum((real -1)**2, dim = 1)
            fake_loss = torch.sum((fake)**2, dim = 1) 
        return 0.5 * torch.mean(real_loss + fake_loss), [torch.mean(real_loss), torch.mean(fake_loss)] 
    def Loss_Generator(self, x, epoch = 1):
        if epoch is not None:
            annealing = min(0.1, epoch / 20)
        else: annealing = 0.1
        
        sample_size = x.shape[0]
        x_gen, set_latent_samples, logdet = self.G.generation(sample_size)
        z, z0, c, c_logit = set_latent_samples

        gen = self.D(x_gen)
        q_code_mu, q_code_logvar = self.Q(x_gen)
        # Generator loss
        if gen.dim() == 2:
            gen_loss = 0.5 * torch.mean((gen - 1)**2)    
        elif gen.dim() == 3:
            gen_loss = 0.5 * torch.mean(torch.sum((gen - 1)**2, dim = 1))    
        if self.c_type == "continuous":
            if self.code_posterior_param == "soft":
                NLL_loss_Q = self.prior_c.NLL_gau(c, q_code_mu, q_code_logvar, "mean")
                
            elif self.code_posterior_param == "hard":
                NLL_loss_Q = self.prior_c.hard_fitting(c, q_code_mu)
            
        elif self.c_type == "discrete":
            # CE
            NLL_loss_Q = self.prior_c.cross_entropy_loss(c_logit if c_logit is not None else c, q_code_mu)
        
        log_qz0 = self.unit_MVG_Guassian_log_prob(z0)
        log_qz0 -= logdet
        log_p_z = self.unit_MVG_Guassian_log_prob(z)
        
        reverse_kl = torch.mean(log_qz0) - torch.mean(log_p_z)
        return gen_loss + (0.2 * NLL_loss_Q) + (annealing * reverse_kl ), [gen_loss, NLL_loss_Q, annealing * reverse_kl]
    
    def unit_MVG_Guassian_log_prob(self, sample):
        return -0.5*torch.sum((sample**2 + np.log(2*np.pi)), dim=1)
class VAE(nn.Module):
    def __init__(self, 
                dx:int = 14, dz:int = 10, dc:int = 6, window:int = 25, d_model:int = 128, 
                num_heads:int = 4, z_projection:str = "aggregation", D_projection:str = "aggregation", 
                time_emb:bool = True, c_type:str = "discrete", c_posterior_param:str = "soft", encoder_E:str = "transformer",
                device = "cpu", supervised = True, kl_annealing = 0.05): 
        super(VAE, self).__init__()
        self.device = device
        self.supervised = supervised
        self.prior_z = md.DiagonalGaussian(dz, mean = 0, var = 1, device=device)
        self.kl_annealing = kl_annealing
        self.dc = dc
        # NF transform
        self.h = md.LatentFlow(dz, self.prior_z)
        self.shared_encoder_layers = None 
        self.c_type = c_type
        self.f_E = md.LatentEncoder(dx=dx, dz=dz, window=window, d_model=d_model, 
                                    num_heads=num_heads, z_projection=z_projection, 
                                    time_emb=time_emb, encoder_E=encoder_E, p_h=self.h) 
        if not supervised:
            self.f_C = None
            self.prior_c = None 
        else: 
            self.prior_c = md.UniformDistribution(device=device) if c_type == "continuous" \
                            else md.DiscreteUniform(dc, onehot = True, device=device)  
            # Since ground truth for c available, f_C is not neccesarily instantiated and trained to capture c
            self.f_C = None
        # Decoder
        self.f_D = md.Decoder(dx=dx, dz=dz, dc=dc, window=window, d_model=d_model,
                                    num_heads=num_heads, p_h = self.h, p_c = None)
        
    def Loss_VAE(self, x, tidx, cond =None, epoch = None):
        if epoch is not None:
            annealing = min(self.kl_annealing, epoch / 15)
        else: annealing = self.kl_annealing
        N, L, _ = x.shape
        # beta is a warm-up coefficient for preventing negative 
        # Encoding
        mu, log_var, zin = self.f_E(x, tidx)
        z, logdet, z0 = self.f_E.reparameterization_NF(mu, log_var)
        # Decoding
        x_rec = self.f_D(z, c = cond, zin = zin)
        
        # Reconstruction & CC is implicitly made since G and f_D share the parameters
        recon = F.mse_loss(x_rec, x, reduction = 'sum') / (N)  # smooth_l1_loss 

        kl_div = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1)).mean()
        
        return recon + (annealing*kl_div), [recon, annealing*kl_div] 

    def L_R(self, x, tidx, cond =None, epoch = None):
        if epoch is not None:
            annealing = min(0.1, epoch / 30)
        else: annealing = 0.1
        N, L, _ = x.shape
        # beta is a warm-up coefficient for preventing negative 
        # Encoding
        mu, log_var, zin = self.f_E(x, tidx)
        # z, logdet, z0 = self.f_E.reparameterization_NF(mu, log_var)
        
        eps = self.prior_z.sample(mu.shape)
        stds = torch.exp(0.5 * log_var)
        z0 = eps * stds + mu
        z, logdet, z0 = self.f_E.p_h(z0 = z0)

        # Decoding
        x_rec = self.f_D(z,c = cond, zin = zin, generation = True)
        
        # # Reconstruction & CC is implicitly made since G and f_D share the parameters
        recon = F.mse_loss(x_rec, x, reduction = 'sum') / (N)  # smooth_l1_loss 

        log_qz0 = self.unit_MVG_Guassian_log_prob(eps) # equiv
        log_qz0 -= torch.sum(0.5 * log_var, dim=1)
        log_qz0 -= logdet
        log_p_z = self.unit_MVG_Guassian_log_prob(z)
        kl_div = torch.mean(log_qz0) - torch.mean(log_p_z)
        
        return recon + (annealing*kl_div) , [recon, annealing*kl_div]
    
    def unit_MVG_Guassian_log_prob(self, sample):
        return -0.5*torch.sum((sample**2 + np.log(2*np.pi)), dim=1)
    
    def stationarization(self, x, tidx, fixed_code =None):
        '''
        This is simply reconstruction in case of unconditional vae
        '''
        mu, log_var, zin = self.f_E(x, tidx)
       
        # Fixed code
        N_c = x.shape[0:-1] + (self.dc,)
        
        if self.supervised:
            if fixed_code:
                target = fixed_code
            else:
                target = 0.1 if self.c_type == "continuous" else 1
            fixed_c = self.prior_c.sample(N_c, target = target)
        else: fixed_c = None
                
        # Decoding
        x_rec = self.f_D(mu,c = fixed_c, zin = zin)
        return x_rec, None
class AE(nn.Module):
    def __init__(self, 
                dx:int = 14, dz:int = 10, dc:int = 6, window:int = 25, d_model:int = 128, 
                num_heads:int = 4, z_projection:str = "aggregation", D_projection:str = "aggregation", 
                time_emb:bool = True, c_type:str = "discrete", c_posterior_param:str = "soft", encoder_E:str = "transformer",
                device = "cpu"): 
        super(AE, self).__init__()
        self.prior_z = md.DiagonalGaussian(dz, mean = 0, var = 1, device=device)
        self.prior_c = None 
           
        self.f_E = md.LatentEncoder(dx=dx, dz=dz, window=window, d_model=d_model, 
                                    num_heads=num_heads, z_projection=z_projection, 
                                    time_emb=time_emb, encoder_E=encoder_E, p_h=None) 
        
        # Decoder
        self.f_D = md.Decoder(dx=dx, dz=dz, dc=dc, window=window, d_model=d_model,
                                    num_heads=num_heads, p_h = None, p_c = None)
        
    def Loss_AE(self, x, tidx, cond = None, epoch = None):
        if epoch is not None:
            annealing = min(.1, epoch / 20)
        else: annealing = 0.1
        # beta is a warm-up coefficient for preventing negative 
        # Encoding
        mu, log_var, zin = self.f_E(x, tidx)
        # Decoding
        x_rec = self.f_D(mu, c = cond, zin = zin)
        
        # Reconstruction & CC is implicitly made since G and f_D share the parameters
        recon = F.mse_loss(x_rec, x, reduction = 'mean') # smooth_l1_loss 
        
        return recon, [recon] 