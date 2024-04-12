import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.att = nn.MultiheadAttention(dim,4,0.2)
        self.w_q = nn.Linear(dim,dim)
        self.w_k = nn.Linear(dim,dim)
        self.w_v = nn.Linear(dim,dim)
        
    def forward(self, x, padding_mask=None):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        return self.att(q,k,v, key_padding_mask=padding_mask)[0]

class TransformerBlock(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.att = SelfAttention(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
            nn.Dropout(0.2)
        )
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
        
    def forward(self, x, padding_mask=None):
        inputs = x
        x = self.att(self.norm1(x), padding_mask)
        x = self.dropout1(x)
        x = x + inputs
        
        att_out = x
        x = self.mlp(self.norm2(x))
        x = self.dropout2(x)
        return att_out + x 
    
class Transformer(nn.Module):
    def __init__(self, hidden_dim, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim) for _ in range(num_blocks)])
    
    def forward(self, x, padding_mask=None):
        for block in self.blocks:
            x = block(x, padding_mask)
        return x, None