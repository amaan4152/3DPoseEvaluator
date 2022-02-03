FROM python:3.8-slim AS base
RUN apt-get update -y && \
    apt-get install make -y && \
    apt-get install python3-opencv -y
RUN apt-get install git -y
RUN apt-get install build-essential gcc libssl-dev zlib1g-dev -y && \
    apt-get install libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev -y && \
    apt-get install libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev openssl -y
RUN curl https://pyenv.run | bash && mv /root/.pyenv /
RUN pip3 install poetry
RUN poetry config virtualenvs.in-project false

FROM base as poetry-build
WORKDIR /root/build
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-dev

FROM poetry-build as proj-config
COPY --from=poetry-build /build/ /build/
WORKDIR /root/build
COPY Makefile ./
COPY src src/
COPY videos videos/
COPY data data/
RUN mkdir output

FROM proj-config as add-gast
COPY --from=poetry-build /build/ /build/
WORKDIR /root/build
RUN git clone https://github.com/fabro66/GAST-Net-3DPoseEstimation.git

FROM proj-config as add-vibe
COPY --from=base /.pyenv/ /.pyenv/
COPY --from=poetry-build /build/ /build/
ENV PATH="/.pyenv/bin:$PATH"
SHELL ["/bin/bash", "-c"]
RUN eval "$(pyenv init -)" && \
    eval "$(pyenv virtualenv-init -)"
RUN pyenv install 3.7.0 && pyenv virtualenv 3.7.0 vibe-env
WORKDIR /root/build
RUN git clone https://github.com/mkocabas/VIBE
WORKDIR /build/VIBE
RUN pyenv local vibe-env && pyenv versions

FROM proj-config as test
COPY --from=proj-config /build/ /build/
WORKDIR /root/build
RUN poetry run make all MODEL=ALL START=600 END=1500

