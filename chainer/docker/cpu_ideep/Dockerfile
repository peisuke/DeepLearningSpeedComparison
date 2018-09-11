FROM ubuntu:16.04 

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    wget \
    sudo \
    vim \
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --upgrade pip

RUN pip3 install numpy \
    pandas \
    matplotlib \
    pillow \
    tqdm \
    chainer==4.4.0 \
    ideep4py==2.0.0.post1
