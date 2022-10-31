import torch.nn as nn
import torch
import torch.nn.functional as F
from .utils import ConvLNBlock

class DurationPredictor(nn.Module):
    def __init__(self, hp, hidden_size, dropout, n_layers):
        super().__init__()
        self.in_linear = nn.Linear(hp.hidden_size + hp.feature_size, hidden_size)
        self.layers = nn.ModuleList([ConvLNBlock(hidden_size, dropout, dilation=(2*i+1)) for i in range(n_layers)])
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, rv, x):
        #x: N, T, C
        #rv, acc: N, C
        t = x.size(1)
        rv = rv.unsqueeze(1).expand(-1, t, -1)
        x = torch.cat([x, rv], 2)
        x = self.in_linear(x).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2) #N, T, C
        x = self.linear(x).squeeze(-1) #N, T
        return x

class F0Predictor(nn.Module):
    def __init__(self, hp, hidden_size, dropout, n_layers, n_outputs=1):
        super().__init__()
        self.in_linear = nn.Linear(hp.hidden_size + hp.feature_size, hidden_size)
        self.layers = nn.ModuleList([ConvLNBlock(hidden_size, dropout, dilation=(2*i+1)) for i in range(n_layers)])
        self.linear = nn.Linear(hidden_size, n_outputs)

    def forward(self, rv, x):
        #x: N, T, C
        #rv, acc: N, C
        t = x.size(1)
        rv = rv.unsqueeze(1).expand(-1, t, -1)
        x = torch.cat([x, rv], 2)
        x = self.in_linear(x).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2) #N, T, C
        feats = x
        x = self.linear(x).squeeze(-1) #N, T
        return x, feats

class EPredictor(nn.Module):
    def __init__(self, hp, hidden_size, dropout, n_layers, n_outputs):
        super().__init__()
        self.hp = hp
        self.in_linear = nn.Linear(hp.f0_hidden_size, hidden_size)
        self.layers = nn.ModuleList([ConvLNBlock(hidden_size, dropout, dilation=(2*i+1)) for i in range(n_layers)])
        self.linear = nn.Linear(hidden_size, n_outputs)

    def forward(self, x, mel_length=None):
        #x: N, T, C
        t = x.size(1)
        x = self.in_linear(x).transpose(1, 2)
        if mel_length is None: #Use approximation during inference
            mel_length = int(t * self.hp.scale_factor)
        x = F.interpolate(x, size=mel_length)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2) #N, T, C
        x = self.linear(x).squeeze(-1) #N, T
        return x


