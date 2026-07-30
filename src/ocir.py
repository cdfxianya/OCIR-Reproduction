import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np
import src.modules as md # (encoders, generators, discriminator_Q, distributions, flow_transforms) 
from src.blocks import transformers

'''TODO: 
1) visualization, metrics

    check dataloading (& full framework once on single minibatch) 

    rul estimator and task 

    ** check data loading first then! 
    ** check full framework
    ** implement visualization
    ** Think about and Implemt rul estimator 

2) loading pipeline 

    trajectory model and task




'''
class OCIR(nn.Module):
    def __init__(self, 
                dx:int = 14, dz:int = 10, dc:int = 6, window:int = 25, d_model:int = 128, 
                num_heads:int = 4, z_projection:str = "aggregation", D_projection:str = "aggregation", 
                time_emb:bool = True, c_type:str = "discrete", c_posterior_param:str = "soft", encoder_E:str = "transformer",
                c_kl = False,
                device = "cpu"): 
        super(OCIR, self).__init__()
        self.device = device
        self.c_type = c_type
        self.z_projection = z_projection
        self.time_emb = time_emb
        self.dc = dc
        self.code_posterior_param = c_posterior_param
        self.c_kl = c_kl
        # Prior Distributions p(z') and p(c)
        self.prior_z = md.DiagonalGaussian(dz, mean = 0, var = 1, device=device)
        # TODO there is distributional mismatch if use gasNLL or implement gaussian one for c 
        self.prior_c = md.UniformDistribution(device=device) if c_type == "continuous" \
           else md.DiscreteUniform(dc, onehot = True, device=device) 
                # else distributions.ContinuousCategorical(dc, 
                #                                         gumbel_temperature= 0.4,
                #                                         decay_rate= 0.95,
                #                                         dist="uniform",
                #                                         device=device)
                # else distributions.DiscreteUniform(dc, onehot = True, device=device) 

        # NF transform
        self.h = md.LatentFlow(dz, self.prior_z)
        
        # shared early layers betweew f_E and f_C
        self.shared_encoder_layers = transformers.SharedEncoder(dx=dx, dz=dz, window=window, d_model=d_model, 
                                          num_heads=num_heads,z_projection=z_projection, time_emb=time_emb)
        # Encoders
        self.f_E = md.LatentEncoder(dx=dx, dz=dz, window=window, d_model=d_model, 
                                          num_heads=num_heads, z_projection=z_projection, 
                                          time_emb=time_emb, encoder_E=encoder_E, p_h=self.h, shared_EC= True if self.shared_encoder_layers is not None else False) 
        
        self.f_C = md.CodeEncoder(dx=dx, dc=dc, d_model=d_model, 
                                        c_type=c_type, c_posterior_param=c_posterior_param, 
                                        shared_EC= True if self.shared_encoder_layers is not None else False,
                                        c_kl = c_kl)
        
        # Generator and decoder
        self.G = md.Decoder(dx=dx, dz=dz, dc=dc, window=window, d_model=d_model,
                                    num_heads=num_heads, p_h = self.h, p_c = self.prior_c)
        self.f_D = md.Decoder(dx=dx, dz=dz, dc=dc, window=window, d_model=d_model,
                                    num_heads=num_heads, p_h = self.h, p_c = self.prior_c) #self.G # shared
        
        # Discriminator and Q
        self.shared_net = transformers.shared_transformer(dx=dx, d_model=d_model, window=window, 
                                                         num_heads=num_heads, D_projection=D_projection)
        # self.shared_net = None
        self.D = md.Discriminator(dx=dx, window=window, d_model=d_model,
                                               num_heads=num_heads, D_projection=D_projection,
                                               shared_layer= self.shared_net)
        self.Q = md.CodePosterior(dx=dx, dc=dc, d_model=d_model, c_type=c_type, 
                                               c_posterior_param=c_posterior_param,
                                               shared_layer=self.shared_net)
    
    
    '''
    For our objective function, L = L_R + L_G + L_RG
    it can be doen in two steps 
    
    1) L_R & L_RG(x -> E -> G)
    Since decoder and G share the same parameters, the objective can be further simplified to L_NF + L_VAE
    L_R = logq(x|z) - KL(logq_h(z|x), logp(z) ) 
    
    2) L_G & L_RG ({z,c} -> G -> E)
    where we embedd the objective of L_RG in L_G.
    '''
    
    # Objective functions
    def L_R(self, x, tidx, epoch = None):
        N, L, _ = x.shape
        # Encoding
        if self.shared_encoder_layers is not None:
            h = self.shared_encoder_layers(x, tidx)
            
            hc = h
            if (self.z_projection == "spc") or (self.z_projection == "seq"):
                hc = hc[:,1:,:]
            if self.time_emb:
                hc = hc[:,:-1,:]
        else: 
            h = x
            hc = x
        mu, log_var, zin = self.f_E(h, tidx)
        # z, logdet, z0 = self.f_E.reparameterization_NF(mu, log_var)
        
        eps = self.prior_z.sample(mu.shape)
        stds = torch.exp(0.5 * log_var)
        z0 = eps * stds + mu
        z, logdet, z0 = self.f_E.p_h(z0 = z0)
        
        c, c_logvar = self.f_C(hc)
        
        if self.c_kl:
            c = self.f_C.reparameterization(c, c_logvar)
            
        # Decoding
        x_rec = self.f_D(z,c, zin = zin)
        x_rec_G = self.G(z.detach(),c.detach(), zin = zin)
        # Reconstruction & CC is implicitly made since G and f_D share the parameters
        recon = F.mse_loss(x_rec, x, reduction = 'sum') / (N)
        recon_G = F.mse_loss(x_rec_G, x, reduction = 'sum') / (N)
        # recon_G = None
        '''
        From normalizing flow, h , 
        log q(zk|x) = log q(z0|x) - sum_{k=1}^K log |det(df_k/dz_{k-1})|   , where zk  is z in our paper and z0 is z'
        
        KL(q(zk | x), p(z)) = E_{q(zk|x)}[log q(zk|x) - log p(z)]
                            = E_{q(z0|x)}[log q(z0|x) - sum_{k=1}^K log |det(df_k/dz_{k-1})| - log p(zk)],  with respect to q(z0|x)
                            -> -KL = E_{q(z0|x)}[- log q(z0|x) + sum_{k=1}^K log |det(df_k/dz_{k-1})| + log p(zk)]  --> we want to minimize it in full ELBO form  
        '''        
        log_qz0 = self.unit_MVG_Guassian_log_prob(eps)
        log_qz0 -= torch.sum(0.5 * log_var, dim=1)
        log_qz0 -= logdet
        log_p_z = self.unit_MVG_Guassian_log_prob(z)
        kl_div = torch.mean(log_qz0) - torch.mean(log_p_z)
        
        if self.c_kl:
            kl_c = self.mmd_loss(c, self.sample_uniform_prior(c), mode = "kernel")
        else: kl_c = None
        # x 0.5 as the reconstruction is done through G (shared net) as well 
        return recon + kl_div, [recon, kl_div, recon_G, kl_c]
    
    def L_G_discriminator(self, x):
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
    
    def L_G_generator(self, x):
        sample_size = x.shape[0]
        x_gen, set_latent_samples, log_det = self.G.generation(sample_size) 
        z, z0, c, c_logit = set_latent_samples 
        
        gen = self.D(x_gen)
        q_code_mu, q_code_logvar = self.Q(x_gen)
        
        # CC
        if self.shared_encoder_layers is not None:
            h = self.shared_encoder_layers(x_gen)
            hc = h
            if (self.z_projection == "spc") or (self.z_projection == "seq"):
                hc = hc[:,1:,:]
            if self.time_emb:
                hc = hc[:,:-1,:]
        else: 
            h = x_gen
            hc = x_gen
        mu, logvar, _ = self.f_E(h)
        c_gen, c_logvar = self.f_C(hc)

        # Generator loss
        if gen.dim() == 2:
            gen_loss = 0.5 * torch.mean((gen - 1)**2)    
        elif gen.dim() == 3:
            gen_loss = 0.5 * torch.mean(torch.sum((gen - 1)**2, dim = 1))    

        # CC and MMI for Q, G  
        cc_loss_z = self.prior_z.NLL(z0, mu, logvar, "mean") # soft fitting
        
        if self.c_type == "continuous":
            if self.code_posterior_param == "soft":
                # NLL
                cc_loss_c = self.prior_c.NLL_gau(c, c_gen, c_logvar, "mean")
                NLL_loss_Q = self.prior_c.NLL_gau(c, q_code_mu, q_code_logvar, "mean")
                
            elif self.code_posterior_param == "hard":
                # MSE
                cc_loss_c = self.prior_c.hard_fitting(c, c_gen)
                NLL_loss_Q = self.prior_c.hard_fitting(c, q_code_mu)
            # One-to-one cycle consistency
            # cc_loss_c = self.prior_c.hard_fitting(c, c_gen)
            
        elif self.c_type == "discrete":
            # CE
            cc_loss_c = self.prior_c.cross_entropy_loss(c_logit if c_logit is not None else c, c_gen)
            NLL_loss_Q = self.prior_c.cross_entropy_loss(c_logit if c_logit is not None else c, q_code_mu)

        return gen_loss + (NLL_loss_Q) + ((cc_loss_z + cc_loss_c)), \
                [gen_loss, NLL_loss_Q , cc_loss_z,  cc_loss_c]
    
    def unit_MVG_Guassian_log_prob(self, sample):
        return -0.5*torch.sum((sample**2 + np.log(2*np.pi)), dim=1)
    def sample_uniform_prior(self, c):
        a = -1.
        b = 1.
        return torch.rand_like(c) * (b - a) + a
    def mmd_loss(self, qc, prior_samples, mode = "mean"):
        """ MMD loss to match q(c) to U(a, b) """
        if mode == "mean":
            return torch.mean((qc.mean(dim=0) - prior_samples.mean(dim=0)) ** 2)
        elif mode == "kernel":
            N, L, dc = qc.shape
            def rbf_kernel(x, y, gamma=None):
                """
                Compute the Gaussian RBF kernel between two tensors x and y.
                """
                N, L, dc = x.shape
                # Compute pairwise squared Euclidean distance
                x_sq = torch.sum(x ** 2, dim=-1, keepdim=True).view(-1, 1)  # Shape: -> (N, 1)
                y_sq = torch.sum(y ** 2, dim=-1, keepdim=True).view(-1, 1)  # Shape: -> (N, 1)
                dist_sq = x_sq - 2 * torch.mm(x.view(-1,dc), y.view(-1,dc).T) + y_sq.T  # Shape: (N, N)
                
                # Default gamma heuristic (median distance)
                if gamma is None:
                    gamma = 1.0 / (2.0 * torch.median(dist_sq))  
                
                return torch.exp(-gamma * dist_sq)
            K_qq = rbf_kernel(qc, qc)  # Kernel matrix for q(c)
            K_pp = rbf_kernel(prior_samples, prior_samples)  # Kernel matrix for p(c)
            K_qp = rbf_kernel(qc, prior_samples) 
            return( K_qq.sum() + K_pp.sum() - 2 * K_qp.sum()) / N# consider sum
    def stationarization(self, x, tidx = None, 
                         fixed_code = None):
        if self.shared_encoder_layers is not None:
            h = self.shared_encoder_layers(x, tidx)
            hc = h
            if (self.z_projection == "spc") or (self.z_projection == "seq"):
                hc = hc[:,1:,:]
            if self.time_emb:
                hc = hc[:,:-1,:]
        else: 
            h = x
            hc = x
        
        mu, log_var, _ = self.f_E(h)
        z, _, _ = self.h(z0 = mu)
        
        # Fixed code
        N_c = x.shape[0:-1] + (self.dc,)

        if fixed_code:
            target = fixed_code
        else:
            target = 0.1 if self.c_type == "continuous" else 1
        fixed_c = self.prior_c.sample(N_c, target = target)
        
        # Reconstruction
        stationarized_X = self.f_D(z, c = fixed_c, generation = True)
        # Generation
        stationarized_X_G = self.G(z, c = fixed_c, generation = True)
        return stationarized_X, stationarized_X_G