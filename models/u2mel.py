import torch.nn as nn
import torch
import torch.nn.functional as F
from .utils import ConvLNBlock, ResBlock

class Unit2Mel(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.source = Generator(hp)
        self.filter = Generator(hp)
        self.energy = EGenerator(hp)
        self.f0_embedding = nn.Embedding(hp.pitch_bins, hp.hidden_size)
        self.unvoiced_embedding = nn.Parameter(torch.randn(hp.hidden_size) * 0.02)
        self.spkr_unit_linear = nn.Linear(hp.hidden_size + hp.feature_size, hp.hidden_size)
        self.spkr_f0_linear = nn.Linear(hp.hidden_size + hp.feature_size, hp.hidden_size)
        self.E_embedding = nn.Embedding(hp.E_bins, hp.hidden_size)

    def encode(self, embedding, bins):
        #bins: N, T, b or N, b
        bins = bins / (bins.sum(-1, keepdim=True) + 1e-5)
        return torch.matmul(bins, embedding.weight)

    def forward(self, E, f0, voicing, spkr, x, mask, mel_length=None, spkr_r=None):
        #x: N, T, C
        #rv, spkr: N, C
        t = x.size(1)
        spkr = spkr.unsqueeze(1).expand(-1, t, -1)
        f0 = self.encode(self.f0_embedding, f0)
        f0[~voicing] = self.unvoiced_embedding.to(f0.dtype)
        E = self.encode(self.E_embedding, E)
        x = self.spkr_unit_linear(torch.cat([x, spkr], 2))
        if spkr_r is not None:
            spkr_r = spkr_r.unsqueeze(1).expand(-1, t, -1)
            f0 = self.spkr_f0_linear(torch.cat([f0, spkr_r], 2))
        else:
            f0 = self.spkr_f0_linear(torch.cat([f0, spkr], 2))
        s = self.source(x, mel_length)
        f = self.filter(f0, mel_length)
        e = self.energy(E)
        ret = s + f
        ret = ret + e
        if mask is not None:
            ret[mask] = 0
        return ret


class Generator(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.layers1 = nn.ModuleList([ConvLNBlock(hp.hidden_size, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers)])
        self.layers2 = nn.ModuleList([ConvLNBlock(hp.hidden_size, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers)])
        self.linear = nn.Linear(hp.hidden_size, hp.n_mels)
        self.hp = hp

    def forward(self, x, mel_length=None):
        #x: N, T, C
        t = x.size(1)
        x = x.transpose(1, 2)
        for layer in self.layers1:
            x = layer(x)
        if mel_length is None: #Use approximation during inference
            mel_length = int(t * self.hp.scale_factor)
        x = F.interpolate(x, size=mel_length)
        for layer in self.layers2:
            x = layer(x)
        x = x.transpose(1, 2) #N, T, C
        x = self.linear(x).squeeze(-1) #N, T, C
        return x

class Discriminator(nn.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.hp = hp
        c_in = hp.n_mels
        c_mid = 256
        c_out = hp.hidden_size

        self.phi = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=3, stride=1, padding=1, dilation=1),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
#            ResBlock(c_mid, c_mid, c_mid),
#            ResBlock(c_mid, c_mid, c_mid),
#            ResBlock(c_mid, c_mid, c_mid),
        )
#        self.res = ResBlock(c_mid, c_mid, c_out)

        self.psi = nn.Conv1d(c_mid, 1, kernel_size=3, stride=1, padding=1, dilation=1)

#        self.match = nn.Sequential(
#            nn.Linear(hp.hidden_size, c_mid),
#            nn.ReLU(),
#            nn.Linear(c_mid, c_mid)
#        )

    def forward(self, mel):
        """
        Args:
            mel: mel spectrogram, torch.Tensor of shape (B x C x T)
            positive: positive speaker embedding, torch.Tensor of shape (B x d)
            negative: negative speaker embedding, torch.Tensor of shape (B x d)
        Returns:
Nsi
        """
        pred1 = self.psi(self.phi(mel))
#        pred = self.res(self.phi(mel))
#        perm = torch.randperm(mel.size(0))
#        pred2 = torch.bmm(spkr.unsqueeze(1), pred)
#        pred3 = torch.bmm(spkr[perm].unsqueeze(1), pred)
#        perm = torch.randperm(mel.size(0))
#        pred4 = torch.bmm(rv.unsqueeze(1), pred)
#        pred5 = torch.bmm(rv[perm].unsqueeze(1), pred)
#        perm = torch.randperm(mel.size(0))
#        pred6 = torch.bmm(acc.unsqueeze(1), pred)
#        pred7 = torch.bmm(acc[perm].unsqueeze(1), pred)
#        pred6 = torch.bmm(self.match(spkr).unsqueeze(1), self.match(rv).unsqueeze(2))
#        pred7 = torch.bmm(self.match(spkr[perm]).unsqueeze(1), self.match(rv).unsqueeze(2))
        result = pred1# + pred2 - pred3 + pred4 - pred5
        result = result.squeeze(1)
        return result#, (pred7 - pred6).squeeze(1).squeeze(1)

class EGenerator(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.layers1 = nn.ModuleList([ConvLNBlock(hp.hidden_size, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers//2)])
        self.linear = nn.Linear(hp.hidden_size, 1)
        self.hp = hp

    def forward(self, x):
        #x: N, T, C
        t = x.size(1)
        x = x.transpose(1, 2)
        for layer in self.layers1:
            x = layer(x)
        x = x.transpose(1, 2) #N, T, C
        x = self.linear(x) #N, T, C
        return x


