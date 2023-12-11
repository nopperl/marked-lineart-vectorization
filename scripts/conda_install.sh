#!/bin/sh
set -euo pipefail

conda install -y -c conda-forge ffmpeg parallel imagemagick
conda install -y -c nvidia cudatoolkit=10.2 cudnn=7
conda install -y cmake
conda install -y -c conda-forge gcc==8.5.0
#conda install -y libgcc
conda install -y pytorch==1.8.1 torchvision==0.9.1 -c pytorch
git clone https://github.com/BachiLi/diffvg --recursive
cd diffvg
git apply ../diffvg.patch
pip install -r ../diffvg-requirements.txt
DIFFVG_CUDA=$FORCE_CUDA CMAKE_PREFIX_PATH=$CONDA_PREFIX python setup.py install
cd ..
pip install -r requirements.txt
mkdir bin \
    && wget https://inkscape.org/gallery/item/26933/Inkscape-c4e8f9e-x86_64.AppImage -O bin/inkscape \
    && chmod u+x bin/inkscape
