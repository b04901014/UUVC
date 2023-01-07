# UUVC

 - Official implementation for the paper [A Unified One-Shot Prosody and Speaker Conversion System with Self-Supervised Discrete Speech Units](http://arxiv.org/abs/2211.06535).
 - Submitted to ICASSP 2023.
 - **Audio samples/demo for our system can be accessed [here](https://b04901014.github.io/UUVC/)**

## Setup environment
1. [pytorch](https://pytorch.org)
2. [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/stable/)
3. [huggingface](https://huggingface.co/docs/transformers/index)
4. [textless-lib](https://github.com/facebookresearch/textlesslib)

See `setup.sh` for details of package installation (especially if you have problem on installing textless-lib).
I used 1.12.1 for pytorch and 1.7.7 for pytorch lightning. No guarantee for the compatability of other versions.

## Get Vocoder
We use the pretrained [HiFi-GAN vocoder](https://github.com/jik876/hifi-gan).
Download the [Universal-V1](https://drive.google.com/drive/folders/1YuOoV3lO2-Hhn1F2HJ2aQ4S0LC1JdKLd) version from the repo and put the checkpoints at `vocoder/cp_hifigan/`

## Dataset preprocessing (for training)
Here we show an example of preprocessing VCTK, which you can adjust to your own dataset.
1. Download VCTK dataset from [here](https://datashare.ed.ac.uk/handle/10283/3443)
2. Run the below commands to preprocess speech units, pitch, 22k melspectrogram, energy, 16k-resampled speech:
```
mkdir -p features/VCTK
python s2u.py --VCTK --datadir VCTK_DIR --outdir features/VCTK --with_pitch_unit
```
The operation is time-consuming due to iterative pitch extraction from textless-lib.

3. Run the below command to generate training and validation splits, you can adjust the script to your own dataset:
```
mkdir datasets
python make_data_vctk.py
```
This will output two file: `datasets/train_vctk.txt` and `datasets/valid_vctk.txt` contains filelists for training and validation.

## Training
We provide the below command as an example, change the arguments according to your dataset:
```
mkdir ckpt
python train.py --saving_path ckpt/ \
                --training_step 70000 \
                --batch_size 200 \
                --check_val_every_n_epoch 5 \
                --traintxt datasets/train_vctk.txt \
                --validtxt datasets/valid_vctk.txt \
                [--distributed]
```
 - `--distributed` if you are training with multiple GPUs
 - `--check_val_every_n_epoch`: Eval every n epoch
 - `--training_step`: Total training step (generator + discriminator)

Tensorboard logging will be in `logs/RV` or `LOG_DIR/RV` if you specify `--logdir LOG_DIR`.

## Inference
We provide examples for synthesis of the system in `inference.py`, you can adjust this script to your own usage.
Example to run `inference.py`:
```
python inference.py --result_dir ./samples --ckpt CKPT_PATH --config CONFIG_PATH --metapath META_PATH
```
 - `--ckpt`: .ckpt file that is generated during training, or from the pretrained checkpoints
 - `--config`: .json file that is generated at the start of the training, or from the pretrained checkpoints
 - `--result_dir`: Your desired output directory for the samples, will create subdirectory for different conversions
 - `--metapath`: The txt file contains the source and target speech paths, see `eval.txt` for an example.

The filenames will be `{source_wav_name}--{target_wav_name}.wav`. For examples of passing original pitch, energy instead of reconstructed, see `inference_exact_pitch.py` with the same arguments.

## Pretrained checkpoints
We provide checkpoints pretrained sperately on VCTK and (LibriTTS-360h + VCTK + ESD). The model is a little bit large since it contains all the training and optimizer states.
 - [VCTK](https://cmu.box.com/s/9w59mb74n97ge18wdfb77htuznmk4y1p)
 - [LibriTTS-360h + VCTK + ESD](https://cmu.box.com/s/76f7kkhuns929da4kaafjqqk2x7nf2d3)

For ethical concerns, the discriminator is also in the checkpoint to distinguish fake from true speech.
