#!/bin/bash -i

MODEL=GAST-Net-3DPoseEstimation
apt-get update
apt-get --no-install-recommends --no-install-suggests -yq install git wget llvm
apt-get install freeglut3 freeglut3-dev ffmpeg -y
git clone https://github.com/fabro66/GAST-Net-3DPoseEstimation.git /root/${MODEL}

cd /root/${MODEL}
pip --no-cache-dir install -U pip
pip --no-cache-dir install -r requirements.txt
rm -f data/video/*  # clear existing videos
rm -f output/*      # clear existing outputs
mkdir checkpoint
cd checkpoint
mkdir yolov3 hrnet gastnet
cd checkpoint/yolov3
wget https://pjreddie.com/media/files/yolov3.weights
mkdir checkpoint/hrnet/pose_coco
cd checkpoint/hrnet/pose_coco
wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS" \
    -O pose_hrnet_w48_384x288.pth && rm -rf /tmp/cookies.txt
cd checkpoint/gastnet
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vh29QoxIfNT4Roqw1SuHDxxKex53xlOB' -O 27_frame_model.bin
