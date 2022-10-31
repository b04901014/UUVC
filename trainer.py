import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.dataset import SpeechDataset
from data.sampler import RandomBucketSampler
from models.s2rv import Speech2Vector
from models.u2mel import Unit2Mel, Discriminator
from models.vp import F0Predictor, DurationPredictor, EPredictor
from models.utils import LengthRegulator
from torch.utils import data
import pytorch_lightning.core.lightning as pl
import soundfile as sf
import librosa
from loss import GANLoss
from vocoder.vocoder import Vocoder
from torch_pitch_shift import *
import random

class BaseTrainer(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.traindata = SpeechDataset(hp, hp.traintxt)
        self.valdata = SpeechDataset(hp, hp.validtxt)

    def load(self):
        state_dict = torch.load(self.hp.pretrained_path)['state_dict']
        self.load_state_dict(state_dict, strict=False)

    def train_dataloader(self):
        sampler = RandomBucketSampler(self.hp.train_bucket_size, self.traindata.lengths, self.hp.batch_size, drop_last=True, distributed=self.hp.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank)
        dataset = data.DataLoader(self.traindata,
                                  num_workers=self.hp.nworkers,
                                  batch_sampler=sampler,
                                  collate_fn=self.traindata.seqCollate)
        return dataset

    def val_dataloader(self):
        dataset = data.DataLoader(self.valdata,
                                  batch_size=self.hp.val_batch_size,
                                  shuffle=True,
                                  num_workers=self.hp.nworkers,
                                  collate_fn=self.valdata.seqCollate)
        return dataset

    def create_optimizer(self, parameters):
        optimizer_adam = optim.Adam(parameters, lr=self.hp.lr, betas=(self.hp.adam_beta1, self.hp.adam_beta2))
        #Learning rate scheduler
        num_training_steps = self.hp.training_step
        num_warmup_steps = self.hp.warmup_step
        num_flat_steps = int(self.hp.optim_flat_percent * num_training_steps)
        def lambda_lr(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step < (num_warmup_steps + num_flat_steps):
                return 1.0
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (num_warmup_steps + num_flat_steps)))
            )
        scheduler_adam = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer_adam, lambda_lr),
            'interval': 'step'
        }
        return optimizer_adam, scheduler_adam

    @property
    def automatic_optimization(self):
        return False

    @property
    def early_stopping_module_freezed(self):
        if self.hp.freeze_module_step is not None:
            return self.global_step > self.hp.freeze_module_step
        return False

    def configure_optimizers(self):
        opt_g, scheduler_g = self.create_optimizer(self.generator_parameters())
        opt_d, scheduler_d = self.create_optimizer(self.discriminator_parameters())
        return [opt_g, opt_d], [scheduler_g, scheduler_d]

class RVectorTrainer(BaseTrainer):
    def __init__(self, hp):
        super().__init__(hp)
        self.RVEncoder = Speech2Vector(hp)
        self.dp = DurationPredictor(hp, hp.dp_hidden_size, hp.dp_dropout, hp.dp_layers)
        self.f0p = F0Predictor(hp, hp.f0_hidden_size, hp.f0_dropout, hp.f0_layers, n_outputs=hp.pitch_bins)
        self.vp = F0Predictor(hp, hp.voiced_hidden_size, hp.voiced_dropout, hp.voiced_layers)
        self.Ep = EPredictor(hp, hp.E_hidden_size, hp.E_dropout, hp.E_layers, hp.E_bins)
        self.u2m = Unit2Mel(hp)
        self.embedding = nn.Embedding(hp.vocab_size+1, hp.hidden_size, padding_idx=hp.vocab_size)
        self.D = Discriminator(hp)
        self.duration_length = LengthRegulator()
        self.gan_loss = GANLoss(1.0, 0.0, 'lsgan')
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        if self.hp.pretrained_path:
            self.load()
        self.vocoder = Vocoder(hp.vocoder_config_path, hp.vocoder_ckpt_path)
        self.vocoder.eval()
        self.vocoder.generator.remove_weight_norm()
        self.shifts = get_fast_shifts(16000)
        for param in self.vocoder.parameters():
            param.requires_grad = False

    def generator_parameters(self):
        return list(self.RVEncoder.parameters()) + list(self.dp.parameters()) + \
            list(self.f0p.parameters()) + list(self.u2m.parameters()) + \
            list(self.embedding.parameters()) + list(self.vp.parameters()) + list(self.Ep.parameters())

    def discriminator_parameters(self):
        return list(self.D.parameters())

    def get_rv_spkr(self, audio, mask):
        return self.RVEncoder(audio, mask)

    def run(self, dedup_unit, unit, rv, spkr, spkr_r, dur, voiced, batch):
        duration_preds = self.dp(dur, dedup_unit)
        f0_preds, f0_feats = self.f0p(rv, unit)
        voiced_preds, _ = self.vp(voiced, unit)
        E = self.Ep(f0_feats, mel_length=batch['mel'].size(1))
        inp_E = (batch['energy'] + torch.sigmoid(E)) / 2
        inp_f0 = (batch['f0'] + torch.sigmoid(f0_preds)) / 2
        pred_mels = self.u2m(inp_E, inp_f0, batch['voiced'].bool(), spkr, unit, batch['mel_mask'], batch['mel'].size(1), spkr_r=spkr_r)
        return pred_mels, duration_preds, f0_preds, voiced_preds, E

    def f0_shift(self, x, mask=None):
        with torch.no_grad():
            ratio = random.choice(self.shifts)
            px = pitch_shift(x.unsqueeze(1), ratio, 16000).squeeze(1).detach()
            if mask is not None:
                px[mask] = 0
            return px

    def forward(self, batch):
        pitch_shifted_audio = self.f0_shift(batch['audio'], batch['audio_mask'])
        rv, speaker_embedding_real, dur, voiced = self.get_rv_spkr(batch['audio'], batch['audio_mask'])
        rv, speaker_embedding, dur, voiced = self.get_rv_spkr(pitch_shifted_audio, batch['audio_mask'])
        #Embed units
        dedup_unit = self.embedding(batch['dedup_unit'])
        unit = self.embedding(batch['unit'])
        pred_mels, duration_preds, f0, voiced_preds, E = self.run(dedup_unit, unit, rv, speaker_embedding, speaker_embedding_real, dur, voiced, batch)
        #Duration prediction
        dur_loss = F.mse_loss(duration_preds[~batch['dedup_unit_mask']], torch.log(1 + batch['duration'][~batch['dedup_unit_mask']].float()))
        #Pitch prediction
        f0_mask = batch['voiced'].bool()
        f0_loss = self.bce_loss(f0[f0_mask], batch['f0'][f0_mask])
        #Voiced prediction
        voiced_loss = self.bce_loss(voiced_preds[~batch['unit_mask']], batch['voiced'][~batch['unit_mask']].float())
        #MelSpec prediction
        mel_loss = F.l1_loss(pred_mels[~batch['mel_mask']], batch['mel'][~batch['mel_mask']])
        #Energy prediction
        E_loss = self.bce_loss(E[~batch['mel_mask']], batch['energy'][~batch['mel_mask']])
        #Embeddings
        E_embed = self.u2m.encode(self.u2m.E_embedding, batch['energy'])
        F_embed = self.u2m.encode(self.u2m.f0_embedding, batch['f0'])
        E_embed_pred = self.u2m.encode(self.u2m.E_embedding, torch.sigmoid(E))
        F_embed_pred = self.u2m.encode(self.u2m.f0_embedding, torch.sigmoid(f0))
        embed_loss = F.mse_loss(E_embed.detach(), E_embed_pred) + F.mse_loss(F_embed.detach(), F_embed_pred)
        mel_loss += embed_loss
        #GAN Loss
        pred_mels = pred_mels.transpose(1, 2)
        pred_gen = self.D(pred_mels)
        gan_loss_g = self.gan_loss(pred_gen, True)
        pred_gen = self.D(pred_mels.detach())
        pred_gt = self.D(batch['mel'].transpose(1, 2).detach())
        gan_loss_d = self.gan_loss(pred_gen, False) + self.gan_loss(pred_gt, True)
        return [E_loss, dur_loss, mel_loss, f0_loss, voiced_loss, gan_loss_g, gan_loss_d]

    def infer(self, batch, shuffled_audio=None, shuffled_mask=None, shuffle_target=None):
        rv, speaker_embedding, dur, voiced = self.get_rv_spkr(batch['audio'], batch['audio_mask'])
        if shuffled_audio is not None:
            if shuffle_target == 'rv':
                rv, _, dur, voiced = self.get_rv_spkr(shuffled_audio, shuffled_mask)
            elif shuffle_target == 'spk':
                _, speaker_embedding, _, _ = self.get_rv_spkr(shuffled_audio, shuffled_mask)
        #Embed units
        dedup_unit = self.embedding(batch['dedup_unit'])
        #Duration prediction
        duration_preds = self.dp(dur, dedup_unit)
        duration_preds = torch.round(torch.exp(duration_preds) - 1)
        duration_preds = torch.clamp(duration_preds, min=1.0) #At least one frame is allocated(?)
        duration_preds[batch['dedup_unit_mask']] = 0.0
        unit = self.embedding(batch['dedup_unit'])
        unit, unit_mask = self.duration_length(unit, duration_preds, None)
        length = int(unit.size(1) * self.hp.scale_factor)
        #Voiced prediction
        voiced_preds, _ = self.vp(voiced, unit)
        voiced = (torch.sigmoid(voiced_preds) > 0.5)
        #Pitch prediction
        f0_preds, f0_feats = self.f0p(rv, unit)
        f0 = torch.sigmoid(f0_preds)
        #Create mask
        pred_mask = F.interpolate((~unit_mask).float().unsqueeze(1), size=length) #TTTT F
        pred_mask = (pred_mask > 0).squeeze(1)
        #Energy prediction
        E = torch.sigmoid(self.Ep(f0_feats))
        #MelSpec Prediction
        pred_mels = self.u2m(E, f0, voiced, speaker_embedding, unit, ~pred_mask)
        return pred_mels, pred_mask

    def training_step(self, batch, batch_idx):
        opt_gen, opt_dis = self.optimizers()
        sch_gen, sch_dis = self.lr_schedulers()
        E_loss, dur_loss, mel_loss, f0_loss, voiced_loss, gan_loss_g, gan_loss_d = self(batch)
        if self.early_stopping_module_freezed:
            loss = mel_loss + gan_loss_g
        else:
            loss = E_loss + dur_loss + mel_loss + f0_loss + voiced_loss + gan_loss_g
        opt_gen.zero_grad()
        self.manual_backward(loss)
        opt_dis.zero_grad()
        self.manual_backward(gan_loss_d)
        opt_gen.step()
        opt_dis.step()
        sch_gen.step()
        sch_dis.step()
        self.log("train/dur", dur_loss, on_step=True, prog_bar=True)
        self.log("train/mel", mel_loss, on_step=True, prog_bar=True)
        self.log("train/f0", f0_loss, on_step=True, prog_bar=True)
        self.log("train/voiced", voiced_loss, on_step=True, prog_bar=True)
        self.log("train/E_loss", E_loss, on_step=True, prog_bar=True)
        self.log("train/G", gan_loss_g, on_step=True, prog_bar=True)
        self.log("train/D", gan_loss_d, on_step=True, prog_bar=True)

    def synthesize(self, batch, shuffle_idx, target):
        vc_mel, mel_mask = self.infer(batch, batch['audio'][shuffle_idx], batch['audio_mask'][shuffle_idx], target)
        vc_mel = vc_mel[0][mel_mask[0]]
        synthetic = self.vocoder(vc_mel.unsqueeze(0)).float()
        return synthetic

    def validation_step(self, batch, batch_idx):
        E_loss, dur_loss, mel_loss, f0_loss, voiced_loss, gan_loss_g, gan_loss_d = self(batch)
        self.log("val/dur", dur_loss, on_epoch=True, logger=True)
        self.log("val/mel", mel_loss, on_epoch=True, logger=True)
        self.log("val/f0", f0_loss, on_epoch=True, logger=True)
        self.log("val/voiced", voiced_loss, on_epoch=True, logger=True)
        self.log("val/E_loss", E_loss, on_epoch=True, logger=True)
        self.log("val/G", gan_loss_g, on_epoch=True, logger=True)
        self.log("val/D", gan_loss_d, on_epoch=True, logger=True)

        #Sample one pair of example of VC
        if batch_idx < self.hp.sample_num:
            original_audio = batch['audio'][0][~batch['audio_mask'][0]]
            rec_mel, mel_mask = self.infer(batch)
            rec_mel = rec_mel[0][mel_mask[0]]
            rec = self.vocoder(rec_mel.unsqueeze(0)).float()
            voc = self.vocoder(batch['mel'][0][~batch['mel_mask'][0]].unsqueeze(0)).float()
            shuffle_idx = torch.randperm(batch['audio'].size(0), device=batch['audio'].device)
            swap_rv = self.synthesize(batch, shuffle_idx, 'rv')
            swap_spk = self.synthesize(batch, shuffle_idx, 'spk')
            #Write files
            sw = self.logger.experiment
            sw.add_audio(f'original/{batch_idx}', original_audio, self.global_step, 16000)
            sw.add_audio(f'condition/{batch_idx}', batch['audio'][shuffle_idx[0]][~batch['audio_mask'][shuffle_idx[0]]], self.global_step, 16000)
            sw.add_audio(f'swap_rv/{batch_idx}', swap_rv, self.global_step, self.hp.sample_rate)
            sw.add_audio(f'swap_speaker/{batch_idx}', swap_spk, self.global_step, self.hp.sample_rate)
            sw.add_audio(f'reconstruct/{batch_idx}', rec, self.global_step, self.hp.sample_rate)
            sw.add_audio(f'vocoder/{batch_idx}', voc, self.global_step, self.hp.sample_rate)
