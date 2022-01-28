FROM busybox
RUN apt install python3 pip && python3 -m pip install --upgrade pip && pip install virtualenv
WORKDIR /build
COPY data/ videos/ src/