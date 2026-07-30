'''
Prior distributions, for p(z') and p(c)
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiagonalGaussian(nn.Module):
    gumbel = False
    def __init__(self, dim, mean = 0., var = 1.,
                 device = "cpu"):
        super(DiagonalGaussian, self).__init__()
        self.dim = dim
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.var = torch.tensor(var, dtype=torch.float32)
        self.std = torch.sqrt(torch.tensor(var, dtype=torch.float32))  # Standard deviation (sqrt of variance)

        self.device =device
    def sample(self, N, 
               mu = None, 
               log_var = None):
        # N shape (*,dz_dim)
        if isinstance(N, tuple):
            N_dim = N
        else: 
            N_dim = N.shape
        eps = torch.randn(N_dim).to(self.device)  # Standard normal random samples
        eps *= torch.exp(0.5 * log_var) if log_var is not None else self.std
        samples = eps + (mu if mu is not None else self.mean)

        return samples

    def log_prob(self, z, mu = None, log_var = None, normalize = True):
        """
        Computes the log probability of a given input x under the Gaussian distribution.
        """
        z_dim = z.shape[-1]

        # Set mean
        if mu is None:
            mu = self.mean
        
        # Set standard deviation
        if log_var is not None:
            log_std = 0.5 * log_var  # log(std) = 0.5 * log_var
        else:
            log_std = torch.log(self.std)  # Store log(std) directly

        diff = z - mu
        log_prob = -0.5 * (diff / torch.exp(log_std)) ** 2  # Quadratic term

        if normalize:
            log_prob -= 0.5 * np.log(2 * np.pi)  # Constant term

        log_prob -= log_std  # Log determinant term

        return log_prob.sum(dim=-1)  # Sum over dimensions
    def H(self, log_var = None):
        """
        Computes the entropy of the diagonal Gaussian distribution.
        """
        # Entropy of a diagonal Gaussian is: H = 0.5 * (dim * log(2 * pi * e) + sum(log(var)))
        if log_var is None:
            log_var = torch.log(self.var)
        entropy = 0.5 * (np.log(2 * np.pi) + torch.sum(log_var, dim = -1))
        return entropy
    
    def NLL(self, z, mu, log_var, aggr = "mean"):
        return -torch.sum(self.log_prob(z, mu, log_var)) if aggr == "sum" else -torch.mean(self.log_prob(z, mu, log_var))

    def hard_fitting(self, z_true, z_mu):
        N = z_true.shape[0]
        return F.mse_loss(z_mu, z_true, reduction = 'sum')  / (N)
    
class UniformDistribution(nn.Module):
    gumbel = False
    def __init__(self, low=-1., high=1.0,
                 device = "cpu"):
        """
        Uniform distribution.
        """
        super(UniformDistribution, self).__init__()
        self.low = low
        self.high = high
        self.device =device
    def sample(self, N, target:float = None):
        # N shape (*,dc_dim)
        if isinstance(N, tuple):
            N_dim = N
        else: 
            N_dim = N.shape
        
        if target is None:
            c = torch.rand(N_dim).to(self.device)
            return c * (self.high - self.low) + self.low
        else:
            if (target > self.high) or (target < self.low):
                target = 0.1
            c = torch.ones(N_dim).to(self.device)
            return c * target
        
    def log_prob(self, c):
        """
        Computes the log probability of a given input c under the uniform distribution.
        """
        # For uniform distribution, log_prob is -log(high - low) if z is within the bounds.
        log_prob = torch.full_like(c, -torch.log(torch.tensor(self.high - self.low, device=self.device)))
        # Check if z is within bounds [low, high]
        log_prob = torch.where((c >= self.low) & (c <= self.high), log_prob, torch.tensor(-float('inf')))
        return log_prob

    def log_prob_gaussian(self, c, mu, log_var, normalize = True):
        """
        This is used for Maximizing mutual information in training Q net for soft fitting 
        Note that this log prob is over gaussian not uniform
        
        """
        c_dim = c.shape[-1]
        log_std = 0.5 * log_var  # log(std) = 0.5 * log_var
        diff = c - mu
        log_prob = -0.5 * (diff / torch.exp(log_std)) ** 2  # Quadratic term

        if normalize:
            log_prob -= 0.5  * np.log(2 * np.pi)  # Constant term

        log_prob -= log_std  # Log determinant term
        return log_prob.sum(dim=-1)  # Sum over dimensions 
    
    def H(self):
        """
        Computes the entropy of the uniform distribution.
        """
        return torch.log(torch.tensor(self.high - self.low, device=self.device))
    def H_gau(self, log_var = None):
        """
        Computes the entropy of the diagonal Gaussian distribution.
        """
        # Entropy of a diagonal Gaussian is: H = 0.5 * (dim * log(2 * pi * e) + sum(log(var)))
        if log_var is None:
            log_var = torch.log(self.var)
        entropy = 0.5 * (np.log(2 * np.pi) + torch.sum(log_var, dim = -1))
        return entropy
    def NLL(self, c, aggr = "sum"): # TODO check this 
        return -torch.sum(self.log_prob(c)) if aggr == "sum" else -torch.mean(self.log_prob(c))
    
    def NLL_gau(self, c, mu, log_var, aggr = "mean"):
        return -torch.sum(self.log_prob_gaussian(c, mu, log_var)) if aggr == "sum" else -torch.mean(self.log_prob_gaussian(c, mu, log_var))
    
    def hard_fitting(self, c_true, c_mu):
        N, L, _ = c_true.shape
        return F.mse_loss(c_mu, c_true, reduction = 'mean') #/ (N+L) 
    
class DiscreteUniform(nn.Module):
    gumbel = False
    def __init__(self, num_classes=10, onehot = False,
                 device = "cpu"):
        """
        Discrete uniform. 
        """
        super(DiscreteUniform, self).__init__()
        self.onehot = onehot
        self.low = 0
        self.num_classes = num_classes  # Number of discrete values
        self.device =device
    def sample(self, N, target = None):
        # N has shape of (*, dc_dim)
        if isinstance(N, tuple):
            N_dim = N[:-1]
        else: 
            N_dim = N.shape[:-1]

        if target is not None:
            if (target > self.num_classes-1) or (target < 0):
                target = 1
            c = torch.ones(N_dim + (1,), dtype = int).to(self.device) * target
        else:
            c = torch.randint(self.low, self.num_classes, N_dim + (1,)).to(self.device) # (*,1)
        
        if self.onehot:
            c = torch.flatten(c, 0, -1)
            c = F.one_hot(c,num_classes = self.num_classes).view(N_dim+(self.num_classes,)) # (*, num_classes)
        else: pass
        return c

    def cross_entropy_loss(self, y, logits): # TODO check later on 
        if self.onehot:
            # We apply softmax to the logits to convert them into probabilities
            probs = F.log_softmax(logits, dim=-1)
            loss = -torch.sum(y * probs, dim=-1)
            return torch.mean(loss)
        else:
            tiny = 1e-7  # for numerical stability
            N = logits.shape[0]
            log_probs = torch.log(logits + tiny)
            
            # Use the integer labels to index the appropriate log probabilities
            log_prob_for_y = log_probs[range(N), y]  # Get log probabilities corresponding to the true classes
            return -torch.mean(log_prob_for_y)  # Negative log likelihood loss (mean over batch)
        
class ContinuousCategorical(nn.Module):
    gumbel = True
    def __init__(self, 
                 num_classes=10,
                 gumbel_temperature = 1.0,
                 decay_rate = 0.95,
                 dist = "normal",
                 device = "cpu"):
        """
        Continuous uniform with a categorical reparameterization trick (Gumbel-softmax is used).
        Used if a explicit logit space of discrete code (differentiable sampling from DiscreteUniform) is desired.  
        """
        super(ContinuousCategorical, self).__init__()
        self.dist =dist
        self.num_classes = num_classes  # Number of discrete values
        self.temperature = gumbel_temperature # smaller the temp, more discrete the distribution is 
        self.decay_rate = decay_rate 
        self.device =device
        raise NotImplementedError("this is under implementation. Use categorical distribution instead")
    def sample(self, N, target:float = None):
        '''Directly sampling logits of discrete variables'''
        # N shape (*,dc_dim)
        if isinstance(N, tuple):
            N_dim = N
        else: 
            N_dim = N.shape
        assert N_dim[-1] == self.num_classes, "The passed N[-1] size does not match the number of classes of the discrete code"
        if self.dist == "uniform": # [0,1]
            if target is None:
                c = torch.rand(N_dim).to(self.device)
            else:
                if (target > 1.) or (target < 0.):
                    target = 0.1
                c = torch.ones(N_dim).to(self.device) * target
        else:
            if target is None:
                c = torch.randn(N_dim).to(self.device)
            else:
                if (target > 1.) or (target < 0.):
                    target = 0.1
                c = torch.ones(N_dim).to(self.device) * target
        logit = c
        return self.gumbel_softmax(logit), logit
    
    def gumbel_softmax(self, logit_c):
        # Sample from Gumbel distribution
        noise = torch.rand_like(logit_c).log().neg().log().to(self.device)
        y = logit_c + noise
        return F.softmax(y / self.temperature, dim=-1)
    
    def step(self):
        # Simple scheduler for temperature
        self.temperature *= self.decay_rate
        if self.temperature <= 0.15:
            self.temperature = 0.15
            
    def cross_entropy_loss(self, logit_true, logit_c):
        log_probs_c = F.log_softmax(logit_c, dim=-1)  # Shape: (batch_size, num_classes)
        target_probs = F.softmax(logit_true, dim=-1)  # Shape: (batch_size, num_classes)
        loss = -torch.mean(torch.sum(target_probs * log_probs_c, dim=-1)) 
        
        return loss   

    def hard_fitting(self, logit_true, logit_c):
        '''Directly computing loss over the logit space is possible without applying softmax and CE'''
        N, L, _ = logit_true.shape
        return F.mse_loss(logit_c, logit_true, reduction = 'mean') # / (N+L) 
