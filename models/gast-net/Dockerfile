FROM python:3.8-slim as base
ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL=GAST-Net-3DPoseEstimation
ENV CMD="python3 gen_skes.py"
ENV OUTPUT_DIR=/root/GAST-Net-3DPoseEstimation/output
ENV DATA_EXT=npz
ENV VIDEO_EXT=mp4
ENV TMP_DIR=/root/tmp
ENV MODE=data
ENV ACTIONS="bash set_device.sh; cp /root/videos/ /root/GAST-Net-3DPoseEstimation/data/video"

COPY requirements.txt .

COPY setup.sh .
RUN setup.sh
RUN mkdir -p /root/${MODEL} ${TMP_DIR}/${MODEL} ${OUTPUT_DIR}
WORKDIR /root/${MODEL}
COPY *.py *.sh ./

ENTRYPOINT [ "bash", "run.sh", "-v -np 1 " ]