import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from copy import deepcopy
import torchaudio
import random

class w2v2(nn.Module): #Small Wrapper
    def __init__(self, m, spk_encoder, dur_encoder):
        super().__init__()
        self.feature_extractor = m.feature_extractor
        self.feature_projection = m.feature_projection
        self.encoder = m.encoder
        self.spk_encoder = spk_encoder
        self.dur_encoder = dur_encoder
        self._get_feature_vector_attention_mask = m._get_feature_vector_attention_mask

class Speech2Vector(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        model = self.get_model()
        spk_encoder = self.get_model().encoder
        spk_encoder.layers = spk_encoder.layers[:hp.spk_layers]
        dur_encoder = self.get_model().encoder
        dur_encoder.layers = dur_encoder.layers[:hp.dur_layers]
        model.encoder.layers = model.encoder.layers[:hp.s2rv_layers]
        self.wav2vec2 = w2v2(model, spk_encoder, dur_encoder)
        self.linear_spk = nn.Linear(model.config.hidden_size, self.hp.feature_size)
        self.linear_dur = nn.Linear(model.config.hidden_size, self.hp.feature_size)
        self.linear = nn.Linear(model.config.hidden_size, self.hp.feature_size)

    def get_model(self):
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        #Simple trick to crop the layers for fine-tuning
        model.feature_extractor.gradient_checkpointing = False
        model.encoder.gradient_checkpointing = False
        model.feature_extractor._freeze_parameters()
        model.config.layerdrop = 0
        return model

    def process(self, x, mask, encoder, linear):
        reps = encoder(x, attention_mask=mask)[0] #N, T, C
        if mask is None:
            rep = reps.mean(1)
        else:
            length = torch.sum(mask.float(), 1, keepdim=True)
            rep = torch.sum(reps, 1) / length
        rep = linear(rep)
        return rep

    def step(self, x, mask=None):
        with torch.no_grad():
            x = self.wav2vec2.feature_extractor(x)
            x = x.transpose(1, 2)
            if mask is not None:
                mask = (~mask).long()
                mask = self.wav2vec2._get_feature_vector_attention_mask(
                    x.shape[1], mask, add_adapter=False
                )
            x, q = self.wav2vec2.feature_projection(x)
            x = x.detach()
        return x, mask

    def forward(self, x, mask=None):
        x, mask = self.step(x, mask)
        spk = self.process(x.clone(), mask, self.wav2vec2.spk_encoder, self.linear_spk)
        rep = self.process(x.clone(), mask, self.wav2vec2.encoder, self.linear)
        dur = self.process(x.clone(), mask, self.wav2vec2.dur_encoder, self.linear_dur)
        vp = rep
        return rep, spk, dur, vp
