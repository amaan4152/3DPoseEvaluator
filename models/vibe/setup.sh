#!/bin/bash -i

apt-get update
apt-get --no-install-recommends -yq install git wget curl unzip llvm freeglut3 freeglut3-dev ffmpeg
git clone https://github.com/mkocabas/VIBE.git /root/VIBE
cd /root/VIBE
pip --no-cache-dir install -U pip
pip --no-cache-dir install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0 joblib
pip --no-cache-dir install git+https://github.com/giacaglia/pytube.git --upgrade
pip --no-cache-dir install -r requirements.txt
pip --no-cache-dir install pyglet==1.5.27
pip --no-cache-dir uninstall -y gdown importlib-metadata
pip --no-cache-dir install gdown
source scripts/prepare_data.sh