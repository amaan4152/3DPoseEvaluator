FROM python:3.6.5-slim as base
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get --no-install-recommends --no-install-suggests -yq install git
SHELL ["/bin/bash", "-c"]
RUN apt-get install unzip llvm freeglut3 freeglut3-dev -y
RUN git clone https://github.com/mkocabas/VIBE.git /root/VIBE
WORKDIR /root/VIBE
RUN pip --no-cache-dir install -U pip && \
    pip --no-cache-dir install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0 && \
    pip --no-cache-dir install git+https://github.com/giacaglia/pytube.git --upgrade && \
    pip --no-cache-dir install -r requirements.txt && \
    pip --no-cache-dir uninstall -y gdown importlib-metadata && \
    pip --no-cache-dir install gdown
RUN source scripts/prepare_data.sh
WORKDIR /home
