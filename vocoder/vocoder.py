import torch.nn as nn
import torch
import json
from .env import AttrDict
from .models import Generator

class Vocoder(nn.Module):
    def __init__(self, config_path, ckpt_path):
        super(Vocoder, self).__init__()
        ckpt = torch.load(ckpt_path)
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.generator = Generator(self.h)
        self.generator.load_state_dict(ckpt['generator'])

    def forward(self, x):
        return self.generator(x.transpose(1, 2))
