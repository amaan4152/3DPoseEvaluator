FROM python:3.8-slim as base
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN pip --no-cache-dir install -U pip && \
    pip --no-cache-dir install poetry
WORKDIR /root
COPY pyproject.toml ./
RUN poetry config virtualenvs.create false
RUN poetry install --only main && pip cache purge
RUN mkdir -p /root/output/gnd_truth
COPY *.py *.sh *.yml ./

ENTRYPOINT [ "bash", "run.sh" ]