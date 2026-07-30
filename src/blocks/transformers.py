import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.blocks import tcns
from src.blocks import src_utils
class SinCosPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        # Create position indices (shape: [max_len, 1])
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)

        # Compute the divisors for sine and cosine terms
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))

        # Compute the positional encodings
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0) 

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each is (B, L, num_heads, head_dim)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, C)
        return self.out_proj(out)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim )
        self.fc2 = nn.Linear(hidden_dim, embed_dim )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.leaky_relu_(self.fc1(x))))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1, prenorm=True):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, ff_hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if self.prenorm:
            x = x + self.dropout(self.attn(self.norm1(x), mask))
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout(self.attn(x, mask)))
            x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class Aggregation(nn.Module):
    # TODO check this 
    def __init__(self, d_model, method='mean'):
        super().__init__()
        self.method = method
        if method == 'weighted':
            self.attn_weights = src_utils.Linear(d_model, 1)  # Learnable attention scores
    def forward(self, hidden_states):
        if self.method == 'mean':
            return hidden_states.mean(dim=1)  # Mean pooling
        elif self.method == 'max':
            return hidden_states.max(dim=1)[0]  # Max pooling
        elif self.method == 'weighted':
            attn_scores = torch.softmax(self.attn_weights(hidden_states), dim=1)  # Compute weights
            return torch.sum(attn_scores * hidden_states, dim=1)  # Weighted sum

class Aggregation_all(nn.Module):
    # TODO check this 
    def __init__(self, d_model, window):
        super().__init__()
        self.aggregation_layer = nn.Linear(d_model * window, d_model)
    def forward(self, z):
        N, L, d = z.shape
        z = torch.flatten(z,1, -1)
        return self.aggregation_layer(z)
        
class Conv1by1(nn.Module):
    # TODO check this 
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, d_model, kernel_size=1)

        if self.conv.weight is not None:
            pass
            init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            init.zeros_(self.conv.bias)
    def forward(self, x):
        # x (N, L, c)
        x = x.permute(0,2,1)
        return self.conv(x).permute(0,2,1)

class shared_transformer(nn.Module):
    def __init__(self,dx:int, d_model:int, window:int, num_heads:int,
                 D_projection:str):
        super(shared_transformer, self).__init__()
        self.num_heads = num_heads
        self.depth = 1
        self.D_projection = D_projection
        self.pos_enc = SinCosPositionalEncoding(d_model, window + 2)
        self.fE_projection = Conv1by1(dx, d_model)
        transformer_encoder = [TransformerEncoderBlock(embed_dim = d_model, num_heads = self.num_heads,
                                                                    ff_hidden_dim = int(d_model * 3), dropout = 0.1,
                                                                    prenorm = True) for _ in range(self.depth)]
        self.transformer_encoder = nn.ModuleList(transformer_encoder)
        
        
        if D_projection == "spc":
            # BERT-style special token (compressive token) is done as discriminative score for sequence
            self.score_token = nn.Parameter(torch.randn(1,1,d_model) * 0.02)

    def forward(self, x, c2_spc = None):
        N = x.shape[0]
        x_emb = x
        x_emb = self.fE_projection(x_emb)
        if (self.D_projection == "spc"):
            score_token = self.score_token.expand(N,-1,-1)
            x_emb = torch.cat((score_token, x_emb), dim = 1)    
        if c2_spc is not None:
            c2_spc = c2_spc.expand(N,-1,-1)
            x_emb = torch.cat((x_emb, c2_spc), dim = 1) 
        # positional encoding
        x_emb = self.pos_enc(x_emb)
        for layer in self.transformer_encoder:
            x_emb = layer(x_emb)
        return x_emb

class SharedEncoder(nn.Module):
    def __init__(self, dx:int, dz:int, window:int, d_model:int, num_heads:int,
                 z_projection:str, time_emb:bool, c2_projection = None
                 ):
        super(SharedEncoder, self).__init__()
        self.z_projection = z_projection
        self.num_heads = num_heads
        self.depth = 1
        self.time_emb = time_emb
        self.c2_projection = c2_projection
        
        if self.time_emb:
            self.time_embedding = nn.Embedding(num_embeddings= 550, embedding_dim = d_model)
            # src_utils.init_embedding(self.time_embedding)
        self.pos_enc = SinCosPositionalEncoding(d_model, window + 2 if c2_projection != "spc" else window + 3)
        self.fE_projection = src_utils.Linear(dx, d_model) #transformers.Conv1by1(dx, d_model)
        
        
        TransformerEncoder = [TransformerEncoderBlock(embed_dim = d_model, num_heads = self.num_heads,
                                                        ff_hidden_dim = int(d_model * 3), dropout = 0.25,
                                                        prenorm =True) for _ in range(self.depth)]
        self.TransformerEncoder = nn.ModuleList(TransformerEncoder)
        
        if z_projection == "spc":
            # BERT-style special token (compressive token) is done as discriminative score for sequence
            self.compressive_token = nn.Parameter(torch.randn(1,1,d_model))
        elif z_projection == "seq":
                self.compressive_token = nn.Parameter(torch.randn(1,1,d_model))
                self.projection_zin = src_utils.Linear(d_model, dz )
                
        if c2_projection == "spc":
            # BERT-style special token (compressive token) is done as discriminative score for sequence
            self.fault_token = nn.Parameter(torch.randn(1,1,d_model))
    def forward(self, x, tidx = None):
        N, L, _ = x.shape
        x_proj = self.fE_projection(x)
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
        if self.z_projection == "spc" or (self.z_projection == "seq"):
            compressive_token = self.compressive_token.expand(N,-1,-1) # TODO repeat?
            # compressive_token = self.compressive_token.repeat(N,1,1) # TODO repeat?

            x_emb = torch.cat((compressive_token, x_emb), dim = 1)  
        
        if self.c2_projection == "spc" :
            fault_token = self.fault_token.expand(N,-1,-1) # TODO repeat?
            # compressive_token = self.compressive_token.repeat(N,1,1) # TODO repeat?

            x_emb = torch.cat((x_emb, fault_token), dim = 1)              
        # positional encoding
        x_emb = self.pos_enc(x_emb)
        
        # transformer
        for layer in self.TransformerEncoder:
            x_emb = layer(x_emb)
            
        return x_emb