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
    python3-numpy \
    python3-wheel \  
    unzip \
    curl

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
    conda clean -tipsy

RUN pip install setuptools 
RUN pip install \
    cython \
    pillow \
    scipy \
    matplotlib \
    pandas \
    h5py \
    tqdm

RUN pip install tensorflow
