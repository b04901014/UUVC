import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F

import argparse
from tqdm import tqdm
import json

from textless.data.speech_encoder import SpeechEncoder
from models.s2rv import Speech2Vector
from models.u2mel import Unit2Mel
from models.vp import F0Predictor, DurationPredictor, EPredictor
from models.utils import LengthRegulator

from pathlib import Path
import random
import pyloudnorm as pyln
from vocoder.vocoder import Vocoder
import os
import soundfile as sf

meter = pyln.Meter(22050)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

parser = argparse.ArgumentParser()

parser.add_argument('--metapath', type=str, required=True)
parser.add_argument('--result_dir', type=str, default='./result')
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--config', type=str, required=True)

args = parser.parse_args()

transform = torchaudio.transforms.Resample(16000, 22050).cuda()

mel_basis = {}
hann_window = {}
def get_energy(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False, return_energy=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    stft = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    stft = torch.sqrt(stft.pow(2).sum(-1)+(1e-9))

    energy = torch.norm(stft, dim=1)
    return energy


class Tester(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.RVEncoder = Speech2Vector(hp)
        self.dp = DurationPredictor(hp, hp.dp_hidden_size, hp.dp_dropout, hp.dp_layers)
        self.f0p = F0Predictor(hp, hp.f0_hidden_size, hp.f0_dropout, hp.f0_layers, n_outputs=hp.pitch_bins)
        self.vp = F0Predictor(hp, hp.voiced_hidden_size, hp.voiced_dropout, hp.voiced_layers)
        self.Ep = EPredictor(hp, hp.E_hidden_size, hp.E_dropout, hp.E_layers, hp.E_bins)
        self.u2m = Unit2Mel(hp)
        self.embedding = nn.Embedding(hp.vocab_size+1, hp.hidden_size, padding_idx=hp.vocab_size)
        bin_size = (hp.f0_max - hp.f0_min) / hp.pitch_bins
        self.f0_bins = torch.arange(hp.pitch_bins, dtype=torch.float32) * bin_size + hp.f0_var_min
        bin_size = hp.E_max / hp.E_bins
        self.E_bins = torch.arange(self.hp.E_bins, dtype=torch.float32) * bin_size
        self.duration_length = LengthRegulator()
        self.vocoder = Vocoder(hp.vocoder_config_path, hp.vocoder_ckpt_path)
        self.vocoder.eval()
        self.vocoder.generator.remove_weight_norm()

    def encode(self, audio):
        PVQ, spk, L, _ = self.RVEncoder(audio)
        return {
            'a_p': PVQ,
            'a_s': spk,
            'a_r': L
        }

    def forward(self, U, V, f0, spkr, E):
        UL = self.embedding(U)
        bins = self.f0_bins.unsqueeze(0).expand(f0.size(0), f0.size(1), -1).to(f0.device)
        f0 = f0.unsqueeze(2).expand(-1, -1, bins.size(-1))
        f0 = torch.exp(-(bins - f0) ** 2 / (2 * self.hp.f0_blur_sigma ** 2))
        #MelSpec Prediction
        pred_mels = self.u2m(E, f0, V, spkr, UL, None)
        #Vocoder, LoudNorm
        wav = self.vocoder(pred_mels).squeeze(0).squeeze(0).detach().cpu().numpy()
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -12.0)
        return wav


with open(args.config, 'r') as f:
    hp = Namespace()
    hp.__dict__ = json.load(f)

model = Tester(hp)
model.load_state_dict(torch.load(args.ckpt)['state_dict'], strict=False)
model.cuda()
model.eval()

encoder = SpeechEncoder.by_name(
    dense_model_name='hubert-base-ls960',
    quantizer_model_name='kmeans',
    vocab_size=hp.vocab_size,
    deduplicate=False,
    need_f0=True
).cuda()

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

with open(args.metapath, 'r') as f:
    lines = f.readlines()

folders = ['exact_speaker']
for folder in folders:
    (Path(args.result_dir) / Path(folder)).mkdir(parents=True, exist_ok=True)

def load(src_wav):
    src_wav, sr = torchaudio.load(src_wav)
    assert sr == 16000
    src_wav = src_wav.cuda()
    if src_wav.size(0) != 1:
        src_wav = src_wav.mean(0)
    return src_wav #1, T


for line in tqdm(lines):
    src_wav_n, tgt_wav_n = line.split()
    out_n = Path(src_wav_n).stem + '--' + Path(tgt_wav_n).stem + '.wav'
    src_wav = load(src_wav_n)
    tgt_wav = load(tgt_wav_n)
    transfer = dict()
    with torch.no_grad():
        src_attributes = model.encode(src_wav)
        tgt_attributes = model.encode(tgt_wav)
        src_wav_22k = transform(src_wav)
        out = encoder(src_wav)
        UL = out['units']
        f0 = out['f0']
        voiced = (f0 > 0)
        f0[voiced] = f0[voiced] - f0[voiced].mean()
        f0[~voiced] = -1000
        U, L = torch.unique_consecutive(UL, return_counts=True)
        U, L, UL = U.unsqueeze(0), L.unsqueeze(0), UL.unsqueeze(0)
        energy = get_energy(src_wav_22k, 1025, 80, 22050, 256, 1024, 0, 8000, return_energy=True)
        energy = energy.unsqueeze(1)
        energy = F.interpolate(energy, size=int(UL.size(1) * hp.scale_factor))
        energy = energy.squeeze()
        energy = torch.exp(-(model.E_bins.to(energy.device).repeat(energy.size(0), 1) - energy.unsqueeze(-1)) ** 2 / (2 * hp.E_blur_sigma ** 2)).unsqueeze(0)
        transfer['exact_speaker'] = model(UL, voiced.unsqueeze(0).cuda(), f0.unsqueeze(0).cuda(), tgt_attributes['a_s'], energy)
    for k in transfer.keys():
        sf.write(os.path.join(args.result_dir, k, out_n), transfer[k], 22050)
