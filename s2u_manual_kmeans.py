import argparse
from pathlib import Path
from tqdm import tqdm
import torchaudio
import numpy as np
import os
import traceback
from transformers import AutoModel
from fast_pytorch_kmeans import KMeans
import torch
import random

parser = argparse.ArgumentParser()
parser.add_argument('--n_clusters', type=int, default=200)
parser.add_argument('--model', type=str, default='hubert-base-ls960')
parser.add_argument('--datadir', type=str, required=True)
args = parser.parse_args()
n_samples_needed = args.n_clusters * 10000
model_name = args.model.split('/')[-1]

encoder = AutoModel.from_pretrained(args.model)
encoder.eval()
encoder = encoder.cuda()

wavfiles = [p for p in Path(args.datadir).rglob('*.wav')]
random.shuffle(wavfiles)

#Get continuous S3Rs
reps, length = [], 0
for f in tqdm(wavfiles):
    wav, sr = torchaudio.load(str(f))
    wav = wav.cuda()
    assert sr == 16000
    with torch.no_grad():
        encoded = encoder(wav).last_hidden_state.detach()
    reps.append(encoded.squeeze(0))
    length += encoded.size(1)
    if length > n_samples_needed:
        break

reps = torch.concat(reps, 0)
kmeans = KMeans(n_clusters=args.n_clusters, mode='euclidean', verbose=1)
labels = kmeans.fit_predict(reps)

for f in tqdm(wavfiles):
    wav, sr = torchaudio.load(str(f))
    wav = wav.cuda()
    assert sr == 16000
    with torch.no_grad():
        encoded = encoder(wav).last_hidden_state.detach().squeeze(0)
    units = kmeans.predict(encoded).detach().cpu().numpy()
#    print (str(f.with_suffix('')) + f'-unit-{model_name}-{args.n_clusters}.npy')
    np.save(str(f.with_suffix('')) + f'-unit-{model_name}-{args.n_clusters}.npy', units)
