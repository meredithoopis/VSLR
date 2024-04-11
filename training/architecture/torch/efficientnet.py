import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SE, self).__init__()

        squeeze_channels = min(int(squeeze_channels * se_ratio),1)
        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv1d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = nn.Swish()
        self.se_expand = nn.Conv1d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y
        return y

class ECA(nn.Module):
    def __init__(self, kernel_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y

class MBBlock(nn.Module):
    def __init__(self,input_dim, output_dim, ratio, kernel_size = 15, stride = 1, dropout=0.2, squeeze_type = 'ECA') -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = int(input_dim * ratio)
        self.kernel_size = kernel_size
        
        block = []
        conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_dim,
                        out_channels=self.hidden_dim, 
                        kernel_size=self.kernel_size , 
                        stride=1, padding='same'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU()
            )    
        block.append(conv1)
        
        if squeeze_type == 'ECA':  
            seblock = ECA()
        else:
            seblock = SE(self.hidden_dim, 10)
        
        block.append(seblock)
        
        projection = nn.Sequential(
            nn.Conv1d(in_channels = self.hidden_dim,
                        out_channels = self.output_dim, 
                        kernel_size = self.kernel_size , 
                        stride = stride, padding='same'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(dropout)
        )
        block.append(projection)
        self.block = nn.Sequential(*block)
        
    def forward(self, x):
        if self.input_dim == self.output_dim:
            return x + self.block(x)
        return self.block(x)
        
        
        
        

class EfficientNET1D(nn.Module):
    config = [
        #(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats)
        [128,  32,  7, 1, 1, 1], # 66
        [32,  64,  7, 2, 4,  2], # 33
        [64,  128,  11, 2, 4,  3], # 16
        [128, 192, 11, 1, 4,  1] # 16 
    ]
    def __inti__(self, input_size, drop_connect_rate = 0.2) -> None:
        super(EfficientNET1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=5, stride=1, padding='same')
        blocks = []
        for in_channels, out_channels, kernel_size, stride, expand_ratio, repeats in self.config:
            # drop connect rate based on block index
            drop_rate = drop_connect_rate * (len(blocks) / 7)
            blocks.append(MBBlock(in_channels, out_channels,  expand_ratio, kernel_size, stride, drop_rate))
            for _ in range(repeats-1):
                drop_rate = drop_connect_rate * (len(blocks) / 7)
                blocks.append(MBBlock(out_channels, out_channels,  expand_ratio, kernel_size, 1, drop_rate))
        self.blocks = nn.Sequential(*blocks)
        self.conv_out = nn.Conv1d(self.config[-1][1], input_size, 1, 1, 0)
        self.dropout = nn.Dropout(drop_connect_rate)
    
    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = self.blocks(x)
        x = F.silu(self.conv_out(x))
        x = self.dropout(x)
        return x