import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet import EfficientNET1D
from transformer import Transformer

class EfficentX(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.efficientnet = EfficientNET1D(input_dim, hidden_dim)
        self.sequence = nn.Identity()
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.efficientnet(x)
        x, _ = self.sequence(x)
        last_hidden_state = x[:,-1,:]
        x = self.fc(last_hidden_state)
        return x 
    
class EfficentTransformer(EfficentX):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__(input_dim, hidden_dim, output_dim)
        self.sequence = Transformer(hidden_dim, 2)
    

        
class EfficentLSTM(EfficentX):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__(input_dim, hidden_dim, output_dim)
        self.sequence = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

class EfficentGRU(EfficentX):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__(input_dim, hidden_dim, output_dim)

        self.sequence = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
    
import time    
if __name__ == '__main__':
    model = EfficentTransformer(66, 192, 10)
    x = torch.randn(32, 100, 66)
    y = model(x)
    print(y.size())
    print("EfficentTransformer parameters", sum(p.numel() for p in model.parameters()))
    start = time.time()
    model(x)
    end = time.time()
    print("Running time:", end-start)
    
    model = EfficentLSTM(100, 192, 10)
    x = torch.randn(32, 100, 100)
    y = model(x)
    print(y.size())
    print("EfficentLSTM parameters", sum(p.numel() for p in model.parameters()))
    start = time.time()
    model(x)
    end = time.time()
    print("Running time:", end-start)
    
    model = EfficentGRU(100, 192, 10)
    x = torch.randn(32, 100, 100)
    y = model(x)
    print(y.size())
    print("EfficentGRU parameters", sum(p.numel() for p in model.parameters()))
    start = time.time()
    model(x)
    end = time.time()
    print("Running time:", end-start)