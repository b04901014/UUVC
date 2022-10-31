from pathlib import Path
import os
import random

features = ['features/VCTK']
splits = []
all_f = list(Path('features/VCTK').rglob('*-unit.npy'))
splits += [s.name for s in all_f if random.random() < 0.05]

to_write_train, to_write_valid = [], []

for f in features:
    units = list(Path(f).rglob('*-unit.npy'))
    for u in units:
        if u.name not in splits:
            to_write_train.append(str(u))
        else:
            to_write_valid.append(str(u))


with open('datasets/train_vctk.txt', 'w') as f:
    for a in to_write_train:
        f.write(a + '\n')
with open('datasets/valid_vctk.txt', 'w') as f:
    for a in to_write_valid:
        f.write(a + '\n')
