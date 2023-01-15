#!/bin/bash -i

apt-get update
apt-get --no-install-recommends --no-install-suggests -yq install llvm freeglut3 freeglut3-dev python3-opencv ffmpeg  protobuf-compiler
pip --no-cache-dir install -U pip
pip --no-cache-dir install -r requirements.txt