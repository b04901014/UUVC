import os
from torch.utils import data
import torch
import json
import numpy as np
import soundfile as sf
import random
from pathlib import Path
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import torchaudio

import torch.nn.functional as F

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x, C=1):
    return torch.exp(x) / C

mel_basis = {}
hann_window = {}
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False, return_energy=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    stft = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    stft = torch.sqrt(stft.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], stft)
    spec = dynamic_range_compression(spec)

    if return_energy:
        energy = torch.norm(stft, dim=1)
        return spec, energy
    return spec

class SpeechDataset(data.Dataset):
    def __init__(self, hp, metadata):
        self.hp = hp
        self.units = self.load_dataset(metadata)
        self.units = [str(x) for x in self.units]
        self.data = [x[:-9] + '.wav' for x in self.units]
        self.mels = [x[:-9] + '-mel.npy' for x in self.units]
#        self.energy = [x[:-9] + '-E-normalized.npy' for x in self.units]
        self.energy = [x[:-9] + '-E.npy' for x in self.units]
        self.f0 = [x[:-9] + '-f0.npy' for x in self.units]
        #Assume 32bit fp PCM, 16000 Hz
        self.lengths = [os.path.getsize(f) / (16000. * 4) for f in self.data]
        bin_size = (self.hp.f0_max - self.hp.f0_min) / self.hp.pitch_bins
        self.f0_bins = torch.arange(self.hp.pitch_bins, dtype=torch.float32) * bin_size + self.hp.f0_var_min
        bin_size = self.hp.E_max / self.hp.E_bins
        self.E_bins = torch.arange(self.hp.E_bins, dtype=torch.float32) * bin_size

        #Print statistics:
        l = len(self.data)
        print (f'Total {l} examples, average length {np.mean(self.lengths)} seconds.')

    def load_dataset(self, metadata):
        units = []
        with open(metadata, 'r') as f:
            for line in f.readlines():
                units.append(line.strip())
        return units

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        audio, sampling_rate = torchaudio.load(self.data[i])
        assert sampling_rate == 16000
        audio = audio[0]
        unit = torch.LongTensor(np.load(self.units[i]))
        dedup_unit, duration = torch.unique_consecutive(unit, return_counts=True)
        mel = torch.FloatTensor(np.load(self.mels[i]))
        energy, f0 = torch.FloatTensor(np.load(self.energy[i])), torch.FloatTensor(np.load(self.f0[i]))
        voiced = (f0 != 0)
        f0mean = f0[voiced].mean()
        f0[voiced] = f0[voiced] - f0mean
        f0[~voiced] = -1000
        f0 = torch.exp(-(self.f0_bins.repeat(f0.size(0), 1) - f0.unsqueeze(-1)) ** 2 / (2 * self.hp.f0_blur_sigma ** 2))
        energy = torch.exp(-(self.E_bins.repeat(energy.size(0), 1) - energy.unsqueeze(-1)) ** 2 / (2 * self.hp.E_blur_sigma ** 2))
        return audio, unit, dedup_unit, duration, mel, energy, f0, voiced.long()

    def seqCollate(self, batch):
        output = {
            'audio': [],
            'speaker': [],
            'unit': [],
            'dedup_unit': [],
            'duration': [],
            'mel': [],
            'audio_mask': [],
            'mel_mask': [],
            'unit_mask': [],
            'dedup_unit_mask': [],
            'energy': [],
            'f0': [],
            'voiced': []
        }
        #Get the max length of everything
        m_a, m_u, m_m, m_d = 0, 0, 0, 0
        for audio, unit, dedup_unit, duration, mel, _, _, _ in batch:
            if len(audio) > m_a:
                m_a = len(audio)
            if len(unit) > m_u:
                m_u = len(unit)
            if len(mel) > m_m:
                m_m = len(mel)
            if len(dedup_unit) > m_d:
                m_d = len(dedup_unit)
        #Pad each element, create mask
        for audio, unit, _, dedup_unit, duration, mel, E, f0, voiced in batch:
            #Deal with audio
            audio_mask = torch.BoolTensor([False] * len(audio) + [True] * (m_a - len(audio)))
            audio = F.pad(audio, [0, m_a-len(audio)])
            #Deal with units
            unit_mask = torch.BoolTensor([False] * len(unit) + [True] * (m_u - len(unit)))
            unit = F.pad(unit, [0, m_u-len(unit)], value=self.hp.vocab_size)
            #Deal with deduplicated units
            dedup_unit_mask = torch.BoolTensor([False] * len(dedup_unit) + [True] * (m_d - len(dedup_unit)))
            dedup_unit = F.pad(dedup_unit, [0, m_d-len(dedup_unit)], value=self.hp.vocab_size)
            duration = F.pad(duration, [0, m_d-len(duration)], value=-100)
            #Deal with mels
            mel_mask = torch.BoolTensor([False] * len(mel) + [True] * (m_m - len(mel)))
            mel = F.pad(mel, [0, 0, 0, m_m-len(mel)])
            #Energy, pitch
            E = F.pad(E, [0, 0, 0, m_m-len(E)])
            f0 = F.pad(f0, [0, 0, 0, m_u-len(f0)])
            voiced = F.pad(voiced, [0, m_u-len(voiced)])
            #Aggregate
            output['audio'].append(audio)
            output['unit'].append(unit)
            output['dedup_unit'].append(dedup_unit)
            output['duration'].append(duration)
            output['mel'].append(mel)
            output['unit_mask'].append(unit_mask)
            output['dedup_unit_mask'].append(dedup_unit_mask)
            output['mel_mask'].append(mel_mask)
            output['audio_mask'].append(audio_mask)
            output['energy'].append(E)
            output['f0'].append(f0)
            output['voiced'].append(voiced)
        for k in output.keys():
            output[k] = torch.stack(output[k])
        return output
