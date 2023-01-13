FROM python:3.8-slim as base
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update

RUN pip --no-cache-dir install -U pip && \
    pip --no-cache-dir install PyYAML
WORKDIR /root

ENTRYPOINT [ "python3", "scripts/compose_configurator.py" ]