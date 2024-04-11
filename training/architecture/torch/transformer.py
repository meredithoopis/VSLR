import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

class TransformerBlock(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.att = nn.MultiheadAttention(dim,8,0.2)