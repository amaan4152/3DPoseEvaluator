FROM python:3.8-slim AS base
RUN apt-get update -y && \
    apt-get install make -y && \
    apt-get install python3-opencv -y
RUN pip3 install poetry
RUN poetry config virtualenvs.in-project false

FROM base as poetry-build
WORKDIR /build
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-dev

FROM poetry-build as proj-config
COPY --from=poetry-build /build/ /build/
WORKDIR /build
COPY Makefile ./
COPY src src/
COPY videos videos/
COPY data data/
RUN mkdir output

FROM proj-config as test
COPY --from=proj-config /build/ /build/
WORKDIR /build
RUN poetry run make all MODEL=ALL START=600 END=1500

