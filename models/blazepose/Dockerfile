FROM python:3.8-slim as base
ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL=Blazepose
ENV CMD="python3 main.py"
ENV OUTPUT_DIR=/root/Blazepose/output
ENV DATA_EXT=npz
ENV VIDEO_EXT=mp4
ENV TMP_DIR=/root/tmp
ENV MODE=data

COPY requirements.txt /
SHELL [ "/bin/bash", "-c" ]

COPY setup.sh /
RUN bash /setup.sh
RUN mkdir -p /root/${MODEL} ${TMP_DIR}/${MODEL} ${OUTPUT_DIR}
WORKDIR /root/${MODEL}
COPY *.py *.sh ./

ENTRYPOINT [ "bash", "run.sh", "-v /root/videos/sit_stand_x3-MATLAB_test.mp4 " ]