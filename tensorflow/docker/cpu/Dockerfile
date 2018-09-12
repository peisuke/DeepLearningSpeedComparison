FROM ubuntu:16.04
 
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    wget \
    git \
    automake \
    cmake \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-wheel \  
    python3-setuptools \
    unzip \
    curl && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install setuptools 
RUN pip3 install \
    cython \
    pillow \
    numpy \
    scipy \
    matplotlib \
    pandas \
    h5py \
    tqdm

RUN pip3 install tensorflow==1.10.1
