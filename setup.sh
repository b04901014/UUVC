conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install pytorch-lightning transformers
git clone git@github.com:facebookresearch/textlesslib.git
cd textlesslib
pip install -e .
pip install git+https://github.com/pytorch/fairseq.git@dd106d9534b22e7db859a6b87ffd7780c38341f
pip install librosa pyloudnorm
