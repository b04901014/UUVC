from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from trainer import RVectorTrainer
from pytorch_lightning.plugins import DDPPlugin
import argparse
import json
import os

parser = argparse.ArgumentParser()

#Paths
parser.add_argument('--saving_path', type=str, default='./ckpt')
parser.add_argument('--resume_checkpoint', type=str, default=None)
parser.add_argument('--traintxt', type=str, default='datasets/train_vctk.txt')
parser.add_argument('--validtxt', type=str, default='datasets/valid_vctk.txt')
parser.add_argument('--logdir', type=str, default='./logs')
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--vocoder_config_path', type=str, default='vocoder/cp_hifigan/config.json')
parser.add_argument('--vocoder_ckpt_path', type=str, default='vocoder/cp_hifigan/g_02500000')

#Optimizer
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=float, default=200)
parser.add_argument('--train_bucket_size', type=int, default=32)
parser.add_argument('--val_batch_size', type=float, default=50)
parser.add_argument('--val_bucket_size', type=int, default=10)
parser.add_argument('--valid_split', type=float, default=0.95)
parser.add_argument('--training_step', type=int, default=300000)
parser.add_argument('--freeze_module_step', type=int, default=None)
parser.add_argument('--optim_flat_percent', type=float, default=0.0)
parser.add_argument('--warmup_step', type=int, default=0)
parser.add_argument('--adam_beta1', type=float, default=0.5)
parser.add_argument('--adam_beta2', type=float, default=0.9)

#Architecture
parser.add_argument('--s2rv_layers', type=int, default=1)
parser.add_argument('--spk_layers', type=int, default=1)
parser.add_argument('--dur_layers', type=int, default=1)
parser.add_argument('--dp_layers', type=int, default=2)
parser.add_argument('--E_layers', type=int, default=2)
parser.add_argument('--f0_layers', type=int, default=4)
parser.add_argument('--voiced_layers', type=int, default=2)
parser.add_argument('--mel_layers', type=int, default=8)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--f0_hidden_size', type=int, default=256)
parser.add_argument('--voiced_hidden_size', type=int, default=128)
parser.add_argument('--dp_hidden_size', type=int, default=128)
parser.add_argument('--E_hidden_size', type=int, default=256)
parser.add_argument('--feature_size', type=int, default=256)
parser.add_argument('--no_texture', action='store_true')

#Ablation
parser.add_argument('--ablated_mixed', action='store_true')
parser.add_argument('--continuous_S3R', action='store_true')

#Dropout
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--f0_dropout', type=float, default=0.3)
parser.add_argument('--f0mean_dropout', type=float, default=0.2)
parser.add_argument('--voiced_dropout', type=float, default=0.3)
parser.add_argument('--dp_dropout', type=float, default=0.3)
parser.add_argument('--E_dropout', type=float, default=0.3)

#Trainer
parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
parser.add_argument('--precision', type=str, choices=['16', '32', "bf16"], default=32)
parser.add_argument('--nworkers', type=int, default=16)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--accelerator', type=str, default='ddp')
parser.add_argument('--version', type=int, default=None)
parser.add_argument('--accumulate_grad_batches', type=int, default=1)

#Data
parser.add_argument('--sample_rate', type=int, default=22050)
parser.add_argument('--n_mels', type=int, default=80)
parser.add_argument('--vocab_size', type=int, default=200)
parser.add_argument('--scale_factor', type=float, default=1.73)
parser.add_argument('--pitch_bins', type=int, default=200)
parser.add_argument('--f0_min', type=float, default=50)
parser.add_argument('--f0_max', type=float, default=550)
parser.add_argument('--f0_var_min', type=float, default=-250)
parser.add_argument('--E_bins', type=float, default=200)
parser.add_argument('--f0_blur_sigma', type=float, default=4)
parser.add_argument('--E_blur_sigma', type=float, default=4)
parser.add_argument('--E_max', type=float, default=200)

#Validate
parser.add_argument('--sample_num', type=int, default=4)


args = parser.parse_args()

with open(os.path.join(args.saving_path, 'config.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

fname_prefix = f''

if args.accelerator == 'ddp':
    args.accelerator = DDPPlugin(find_unused_parameters=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=args.saving_path,
    filename=(fname_prefix+'{epoch}-{step}'),
#    every_n_train_steps=(None if args.val_check_interval == 1.0 else args.val_check_interval),
    every_n_epochs=(None if args.check_val_every_n_epoch == 1 else args.check_val_every_n_epoch),
    verbose=True,
    save_last=True
)

if args.continuous_S3R:
    from trainer_cont import RVectorTrainer


logger = TensorBoardLogger(args.logdir, name="RV", version=args.version)

wrapper = Trainer(
    precision=args.precision,
    amp_backend='native',
    callbacks=[checkpoint_callback],
    resume_from_checkpoint=args.resume_checkpoint,
#    val_check_interval=args.val_check_interval,
    num_sanity_val_steps=0,
    max_steps=args.training_step,
    gpus=(-1 if args.distributed else 1),
    strategy=(args.accelerator if args.distributed else None),
    replace_sampler_ddp=False,
    accumulate_grad_batches=args.accumulate_grad_batches,
    logger=logger,
    check_val_every_n_epoch=args.check_val_every_n_epoch
)
model = RVectorTrainer(args)
wrapper.fit(model)
