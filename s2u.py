from torch.utils import data
import argparse
from pathlib import Path
from tqdm import tqdm

import torchaudio
from textless.data.speech_encoder import SpeechEncoder

from data.dataset import mel_spectrogram
import numpy as np
import os
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--n_clusters', type=int, choices=[50, 100, 200], default=200)
parser.add_argument('--model', type=str, default='hubert-base-ls960')
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--with_pitch_unit', action='store_true')
parser.add_argument('--VCTK', action='store_true')

args = parser.parse_args()

if args.with_pitch_unit:
    encoder = SpeechEncoder.by_name(
        dense_model_name=args.model,
        quantizer_model_name='kmeans',
        vocab_size=args.n_clusters,
        deduplicate=False,
        need_f0=False
    ).cuda()

processed = [p.stem for p in Path(args.outdir).glob(f'*-unit.npy')]
transforms_16k = {
    22050: torchaudio.transforms.Resample(22050, 16000).cuda(),
    24000: torchaudio.transforms.Resample(24000, 16000).cuda(),
    48000: torchaudio.transforms.Resample(48000, 16000).cuda()
}
transforms_22k = {
    24000: torchaudio.transforms.Resample(24000, 22050).cuda(),
    16000: torchaudio.transforms.Resample(16000, 22050).cuda(),
    48000: torchaudio.transforms.Resample(48000, 22050).cuda()
}

wavfiles = [p for p in Path(args.datadir).rglob('*.wav')] + [p for p in Path(args.datadir).rglob('*.flac')]
if args.VCTK:
    wavfiles = [p for p in wavfiles if '_mic1' in str(p)]

print (wavfiles)
for f in tqdm(wavfiles):
#    if f.stem + '-unit' in processed:
#        continue
    wav, sr = torchaudio.load(str(f))
    wav = wav.cuda()
    wav_16k = wav
    if sr != 16000:
        if sr not in transforms_16k:
            continue
        wav_16k = transforms_16k[sr](wav)
    if sr != 22050:
        if sr not in transforms_22k:
            continue
        wav = transforms_22k[sr](wav)
    try:
        mels, energy = mel_spectrogram(wav, 1025, 80, 22050, 256, 1024, 0, 8000, return_energy=True)
        mels, energy = mels.cpu().numpy()[0].T, energy.cpu().numpy()[0]
        if args.with_pitch_unit:
            encoded = encoder(wav_16k)
            units = encoded["units"].cpu().numpy()
            f0 = encoded["f0"].cpu().numpy()
    except:
        print (f.stem)
        traceback.print_exc()
        continue
    name = f.stem
    if args.VCTK:
        name = name.replace('_mic1', '')
    np.save(os.path.join(args.outdir, name + '-mel.npy'), mels)
    np.save(os.path.join(args.outdir, name + '-E.npy'), energy)
    torchaudio.save(os.path.join(args.outdir, name + '.wav'), wav_16k.cpu(), 16000)
    if args.with_pitch_unit:
        np.save(os.path.join(args.outdir, name + '-unit.npy'), units)
        np.save(os.path.join(args.outdir, name + '-f0.npy'), f0)
