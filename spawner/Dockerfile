FROM python:3.8-slim as base
ENV DEBIAN_FRONTEND noninteractive
RUN pip --no-cache-dir install -U pip
RUN pip --no-cache-dir install PyYAML

WORKDIR /root/spawner
ENTRYPOINT [ "source", "run.sh" ]
