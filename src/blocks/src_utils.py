import torch
import torch.nn as nn
import torch.nn.init as init
from src.blocks import src_utils

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, 
                 bias=True, init_relu = False, noraml_small = False, zero = False):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.init_relu = init_relu
        self.noraml_small = noraml_small
        self.zero = zero
        self._initialize_weights()
        
    def _initialize_weights(self):
        # TODO: need to set bound
        if self.init_relu:
            nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
            pass
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        elif self.zero:
            nn.init.zeros_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        else:
            if self.noraml_small:
                init.normal_(self.weight, std = 0.01)
            else:
                if self.bias is not None:
                    init.zeros_(self.bias)
                
class rnn_aggregation(nn.Module):
    def __init__(self,input_dim = 10, hidden_dim = 32, 
                 dropout_ = 0., layer = 1):
        super(rnn_aggregation, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.gru = nn.GRU(input_dim, hidden_dim,
                                  num_layers= layer,
                                  batch_first = True,
                                  bias = True,
                                  dropout = dropout_)
        self.apply(src_utils.init_gru)
    def forward(self, x):
        h0 = torch.zeros((self.layer, x.shape[0], self.hidden_dim)).to(x.device)
        _, h_t = self.gru(x,h0)
        return h_t[-1,:,:] # (batch, hidden)
    
class rnn_decoding_eq(nn.Module):
    def __init__(self,dz, dc, 
                 hidden_dim = 32, 
                 dropout_ = 0., layer = 1, 
                 window = 25,
                 learnable = True):
        super(rnn_decoding_eq, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.window = window
        self.learnable = learnable
        
        self.gru = nn.GRU(dz + dc, 
                          hidden_dim,
                            num_layers= layer,
                            batch_first = True,
                            bias = True,
                            dropout = dropout_)
        self.linear_layer = Linear(hidden_dim, hidden_dim, bias = False)
        
        if learnable:
            self.seqtoken = nn.Parameter(torch.randn(1,window, dz))
        else:
            seqtoken = torch.randn(1, window, dz)
            self.register_buffer("seqtoken", seqtoken)        
    def forward(self, z, c = None):
        N,dz = z.shape
        z = z.unsqueeze(1).repeat(1, self.window, 1) # expansion  or expand(N, L, dz)
        seqtokens = self.seqtoken.to(z.device)
        z = z + seqtokens
        
        if c is not None:
            z = torch.concatenate((z,c), dim = -1)
        
        h0 = torch.zeros((self.layer, z.shape[0], self.hidden_dim)).to(z.device)
        z, _ = self.gru(z, h0)
        
        return self.linear_layer(z) # (batch, L, hidden)
    
class rnn_decoding_seqtoken(nn.Module):
    '''Use the input z as starting state to propagate'''
    def __init__(self, dz, dc, hidden_dim = 32, 
                 dropout_ = 0., layer = 1, 
                 learnable = True, window = 100,
                 seq_out = False):
        super(rnn_decoding_seqtoken, self).__init__()# TODO 
        self.hidden_dim = hidden_dim
        self.dz = dz
        self.layer = layer
        self.window = window
        self.seq_out = seq_out
        
        self.lstm = nn.LSTM(dz + dc, hidden_dim,
                                  num_layers= layer,
                                  batch_first = True,
                                  bias = True,
                                  dropout = dropout_)
        
        if not seq_out:
            if learnable:
                self.seqtoken = nn.Parameter(torch.randn(1,window, dz) )
            else:
                seqtoken = torch.randn(1, window, dz)
                self.register_buffer("seqtoken", seqtoken)
        self.expansion = Linear(dz,hidden_dim, bias = False)
    def forward(self, z, c = None, zin = None, c2 = None):
        N, dz = z.shape
        if not self.seq_out:
            seqtoken = self.seqtoken.repeat(N, 1, 1).to(z.device) # expansion
            
            if c is not None:
                c = torch.flip(c, dims = [1]) # flip the seq indices so that the hidden state can influence from the last 
                seqtoken_c = torch.concatenate((seqtoken,c), dim = -1)
            else:
                seqtoken_c = seqtoken
        elif (self.seq_out) and (zin is not None): 
            seqtoken_c = zin
        h0 = self.expansion(z)
        h0 = h0.unsqueeze(0).repeat(self.layer, 1, 1)
        c0 = torch.zeros_like(h0).to(z.device)
        z_c, (_,_) = self.lstm(seqtoken_c, (h0, c0))

        z_c = torch.flip(z_c, dims = [1]) # flip back to the original order
        return z_c

class rnn_decoding(nn.Module):
    '''Use the input z as starting state to propagate'''
    def __init__(self, dz, dc, hidden_dim = 32, 
                 dropout_ = 0., layer = 1, 
                 learnable = True, window = 100):
        super(rnn_decoding, self).__init__()# TODO 
        self.hidden_dim = hidden_dim
        self.dz = dz
        self.layer = layer
        self.window = window
        
        self.lstm = nn.LSTM(dz + dc, hidden_dim,
                                  num_layers= layer,
                                  batch_first = True,
                                  bias = True,
                                  dropout = dropout_)

    def forward(self, z, c = None, zin = None):
        N, dz = z.shape
        z = z.unsqueeze(1).expand(-1, self.window, -1).to(z.device) # expansion
        if c is not None:
            seqtoken_c = torch.concatenate((z,c), dim = -1)
        else:
            seqtoken_c = z
        
        # print(seqtoken_c)
        h0 = torch.zeros((self.layer, z.shape[0], self.hidden_dim)).to(z.device)
        c0 = torch.zeros_like(h0).to(z.device)
        z_c, (_,_) = self.lstm(seqtoken_c, (h0, c0))
        return z_c

class wide_decoding(nn.Module):
    '''Use the input z as starting state to propagate'''
    def __init__(self, dz, dc, hidden_dim = 32, window = 100,
                 dc2 = 0, expansion_dim = 80):
        super(wide_decoding, self).__init__()# TODO 
        self.hidden_dim = hidden_dim
        self.dz = dz
        self.window = window
        self.expansion_dim = expansion_dim
        
        self.expansion = Linear(dz + dc2, expansion_dim * window, bias=False)
        self.expansion2 = Linear(expansion_dim + dc, hidden_dim, bias=False)
        # self.apply(src_utils.init_gru)
    def forward(self, z, c = None, zin = None, c2 = None):
        N, dz = z.shape
        if c2 is not None:
            z = torch.concatenate((z,c2), dim = -1)
        else:
            z = z
        z = self.expansion(z)
        z = z.view(N, self.window, self.expansion_dim)
        z += (torch.randn(z.shape) * 0.1).to(z.device) # TODO 
        # print(z.mean(-1))
        
        if c is not None:
            z = torch.concatenate((z,c), dim = -1)
        else:
            z = z
            
        return self.expansion2(z)

class comb_decoding(nn.Module):
    def __init__(self, dz, dc, hidden_dim = 32, window = 100):
        super(comb_decoding, self).__init__()# TODO 
        self.window = window
        self.expansion = Linear(dz+dc, hidden_dim, bias = False)
    def forward(self, z, c = None, zin = None):
        N, dz = z.shape
        z = z.unsqueeze(1).repeat(1,  self.window, 1) # expansion
        z += (torch.randn(z.shape) * 0.04).to(z.device)
        
        if c is not None:
            seqtoken_c = torch.concatenate((z,c), dim = -1)
        else:
            seqtoken_c = z
        return self.expansion(seqtoken_c)

class LayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))  
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))  

    def forward(self, x):
        # x: (B, C, L)
        mean = x.mean(dim=1, keepdim=True)  
        var = x.var(dim=1, unbiased=False, keepdim=True)  

        x_norm = (x - mean) / torch.sqrt(var + self.eps) 
        return self.gamma * x_norm + self.beta  
    
    
class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()
    def forward(self, input):
        return torch.sin(input)

def init_mlps(m):
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def init_gru(m):
    if isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

def init_embedding(m):
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)