FROM python:3.8-slim as base
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get --no-install-recommends --no-install-suggests -yq install make python3-opencv
RUN apt-get install --no-install-recommends --no-install-suggests -yq build-essential gcc wget curl llvm ffmpeg
WORKDIR /root