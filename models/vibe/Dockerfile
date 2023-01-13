FROM python:3.6.5-slim as base
ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL=VIBE
ENV CMD="python3 demo.py"
ENV OUTPUT_DIR=/root/VIBE/output
ENV DATA_EXT=pkl
ENV VIDEO_EXT=mp4
ENV TMP_DIR=/root/tmp
ENV MODE=data
ENV ACTIONS="bash set_device.sh"

COPY setup.sh .
RUN setup.sh
RUN mkdir -p /root/${MODEL} ${TMP_DIR}/${MODEL} ${OUTPUT_DIR}
WORKDIR /root/${MODEL}
COPY *.py *.sh ./

ENTRYPOINT [ "bash", "run.sh", "--vid_file --output_folder output/ --tracker_batch_size 1 --vibe_batch_size 64 --no_render" ]