FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
	sudo \
        build-essential \
        cmake \
        git \
        wget \
	curl \
        libboost-all-dev \
        libopencv-dev \
        protobuf-compiler \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3-setuptools \
        python3-scipy && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install mxnet tqdm
