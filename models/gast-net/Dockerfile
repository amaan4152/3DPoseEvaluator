FROM python:3.6.5-slim as base
ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL=GAST-Net-3DPoseEstimation
ENV CMD="python3 gen_skes.py"
ENV OUTPUT_DIR=/root/GAST-Net-3DPoseEstimation/output
ENV DATA_EXT=npz
ENV VIDEO_EXT=mp4
ENV TMP_DIR=/root/tmp
ENV MODE=data
ENV ACTIONS="bash set_device.sh && cp /root/video/sit_stand_x3-MATLAB_test.mp4 /root/GAST-Net-3DPoseEstimation/data/video"

COPY requirements.txt /
SHELL [ "/bin/bash", "-c" ]

COPY setup.sh /
RUN bash /setup.sh
WORKDIR /root/${MODEL}
COPY *.py *.sh ./

ENTRYPOINT [ "bash", "run.sh", "-v sit_stand_x3-MATLAB_test.mp4 -np 1 " ]