FROM python:3.8-slim as base
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get --no-install-recommends --no-install-suggests -yq install make python3-opencv git
RUN apt-get install --no-install-recommends --no-install-suggests -yq build-essential gcc libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev openssl ffmpeg
RUN curl https://pyenv.run | bash
RUN pip3 install poetry

FROM base as pyenv-config
COPY --from=base /root/.pyenv/ /root/.pyenv/
COPY poetry.lock pyproject.toml ./
COPY scripts/ /root/scripts
WORKDIR /root/scripts
RUN poetry config virtualenvs.create false
RUN poetry install --only main && pip cache purge
ENV PATH /root/.pyenv/shims:/root/.pyenv/bin:$PATH
SHELL ["/bin/bash", "-c"]
RUN bash -x init_pyenv.sh

# Ref: https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
FROM pyenv-config as add-gast
COPY --from=pyenv-config /root/.bashrc /root/.bashrc
COPY --from=pyenv-config /root/.pyenv /root/.pyenv
SHELL ["/bin/bash", "-c"]
RUN git clone https://github.com/fabro66/GAST-Net-3DPoseEstimation.git /root/GAST-Net-3DPoseEstimation
WORKDIR /root/GAST-Net-3DPoseEstimation
COPY src/models/gast-requirements.txt ./
RUN source ~/.bashrc pyenv && \
    pyenv local gast-env && \
    python3 -m pip --no-cache-dir install -U pip && \
    pip --no-cache-dir install -r gast-requirements.txt && \
    rm -f data/video/* && rm -f output/*
RUN mkdir checkpoint && cd checkpoint && \
    mkdir yolov3 hrnet gastnet
RUN cd checkpoint/yolov3 && wget https://pjreddie.com/media/files/yolov3.weights
RUN mkdir checkpoint/hrnet/pose_coco && \
    cd checkpoint/hrnet/pose_coco && \ 
    wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS" \
    -O pose_hrnet_w48_384x288.pth && rm -rf /tmp/cookies.txt
RUN cd checkpoint/gastnet && \
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vh29QoxIfNT4Roqw1SuHDxxKex53xlOB' -O 27_frame_model.bin


FROM pyenv-config as add-vibe
COPY --from=add-gast /root/.bashrc /root/.bashrc
COPY --from=add-gast /root/.pyenv /root/.pyenv
COPY --from=add-gast /root/GAST-Net-3DPoseEstimation /root/GAST-Net-3DPoseEstimation
WORKDIR /root
SHELL ["/bin/bash", "-c"]
RUN git clone https://github.com/mkocabas/VIBE.git /root/VIBE
WORKDIR /root/VIBE
RUN pyenv local vibe-env
RUN source ~/.bashrc && \
    pip --no-cache-dir install -U pip && \
    pip --no-cache-dir install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0 && \
    pip --no-cache-dir install git+https://github.com/giacaglia/pytube.git --upgrade && \
    pip --no-cache-dir install -r requirements.txt && \
    pip --no-cache-dir uninstall -y gdown importlib-metadata && \
    pip --no-cache-dir install gdown
RUN apt-get install unzip llvm freeglut3 freeglut3-dev -y
RUN source scripts/prepare_data.sh
RUN mkdir /root/Blazepose /root/Blazepose/output 
COPY src/models/exec_blazepose.py /root/Blazepose
WORKDIR /home/

ENTRYPOINT [ "make", "all" ]
