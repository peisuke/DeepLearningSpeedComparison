FROM ubuntu:16.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    wget \
    sudo \
    bzip2 \
    vim && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Configure environment
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN cd /tmp && \
    mkdir -p $CONDA_DIR && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86_64.sh && \
    /bin/bash Miniconda3-4.3.21-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.3.21-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda install --quiet --yes conda==4.3.21 && \
    $CONDA_DIR/bin/conda config --system --add channels conda-forge && \
    $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
    $CONDA_DIR/bin/conda install --quiet --yes \
    pillow \
    cython \
    tqdm && \
    $CONDA_DIR/bin/pip install --upgrade pip && \
    $CONDA_DIR/bin/pip install --ignore-installed -U nnabla && \
    $CONDA_DIR/bin/conda clean -tipsy

