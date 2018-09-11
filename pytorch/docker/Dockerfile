FROM python:3.6

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    wget \
    sudo \
    vim \
    curl && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
   
RUN pip install numpy \
    pandas \
    matplotlib \
    pillow \
    tqdm

RUN pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl && \
    pip install torchvision
