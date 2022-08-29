FROM python:3.8-slim as base
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN pip --no-cache-dir install -U pip && \
    pip --no-cache-dir install mediapipe
RUN mkdir /root/Blazepose /root/Blazepose/output 
WORKDIR /root/Blazepose
COPY exec_blazepose.py ./
WORKDIR /home
