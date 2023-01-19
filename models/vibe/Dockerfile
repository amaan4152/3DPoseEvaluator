FROM python:3.7.0-slim as base
ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL=VIBE
ENV CMD="python3 demo.py"
ENV OUTPUT_DIR=/root/VIBE/output
ENV DATA_EXT=pkl
ENV VIDEO_EXT=mp4
ENV TMP_DIR=/root/tmp
ENV MODE=data
ENV ACTIONS="bash set_device.sh"
SHELL [ "/bin/bash", "-c" ]

COPY setup.sh /
RUN bash /setup.sh
WORKDIR /root/${MODEL}
COPY *.py *.sh ./

ENTRYPOINT [ "bash", "run.sh", "--vid_file /root/video/sit_stand_x3-MATLAB_test.mp4 --output_folder output/ --tracker_batch_size 1 --vibe_batch_size 64 --no_render" ]