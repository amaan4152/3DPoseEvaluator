#!/bin/bash -i
if ! command -v nvidia-smi &> /dev/null
then
    echo "NOTICE: CUDA driver not detected assuming device is CPU..."

    # VIBE uses torch.load to load in pretrained data. However, this causes problems
    # for CPU based devices that execute VIBE. In order to bypass this problem we can 
    # modify  all instances of torch.load(pretrained) and add the special argument map_location='cpu'
    sed -i "s/torch\.load(*/&\, map\_location=\'cpu\'/g" /root/VIBE/*
    exit
fi