#!/bin/bash -i

MODEL=GAST-Net-3DPoseEstimation
apt-get update
apt-get --no-install-recommends --no-install-suggests -yq install git wget curl llvm freeglut3 freeglut3-dev ffmpeg
git clone https://github.com/fabro66/GAST-Net-3DPoseEstimation.git /root/${MODEL}
pip --no-cache-dir install -U pip
pip --no-cache-dir install -r requirements.txt

cd /root/${MODEL} || { echo "ERROR: Directory doesn't exist"; exit 1; }
rm -f data/video/*  # clear existing videos
rm -f output/*      # clear existing outputs
mkdir checkpoint
cd checkpoint || { echo "ERROR: Directory doesn't exist"; exit 1; }
mkdir yolov3 hrnet gastnet
cd yolov3 || { echo "ERROR: Directory doesn't exist"; exit 1; }
wget https://pjreddie.com/media/files/yolov3.weights
cd ..
mkdir hrnet/pose_coco
cd hrnet/pose_coco || { echo "ERROR: Directory doesn't exist"; exit 1; }
wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS" \
    -O pose_hrnet_w48_384x288.pth
rm -rf /tmp/cookies.txt
cd ../..
cd gastnet || { echo "ERROR: Directory doesn't exist"; exit 1; }
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vh29QoxIfNT4Roqw1SuHDxxKex53xlOB' -O 27_frame_model.bin
