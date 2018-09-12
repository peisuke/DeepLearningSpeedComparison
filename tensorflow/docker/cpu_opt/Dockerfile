FROM ubuntu:16.04
 
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    libopenblas-dev \
    git \
    automake \
    cmake \
    pkg-config \
    unzip \
    curl \
    wget \
    libcurl3-dev \
    libfreetype6-dev \
    libpng12-dev \
    libzmq3-dev \
    rsync \
    zip \
    zlib1g-dev \
    libssl-dev \
    openjdk-8-jdk \
    openjdk-8-jre-headless && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PYENV_ROOT="/.pyenv" \
    PATH="/.pyenv/bin:/.pyenv/shims:$PATH"

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash && \
    pyenv update && pyenv install 3.6.6 && pyenv global 3.6.6

RUN pip install setuptools 
RUN pip install \
    cython \
    pillow \
    numpy \
    scipy \
    matplotlib \
    pandas \
    h5py \
    wheel \
    tqdm

# Set up Bazel.
RUN wget --no-check-certificate -q https://github.com/bazelbuild/bazel/releases/download/0.15.2/bazel-0.15.2-dist.zip && \
    unzip bazel-0.15.2-dist.zip -d bazel-0.15.2-dist && \
    chmod -R ug+rwx bazel-0.15.2-dist && \
    cd bazel-0.15.2-dist && \
    ./compile.sh && \
    cp output/bazel /usr/local/bin/ && \
    cd ../ && \
    rm -rf bazel-0.15.2-dist bazel-0.15.2-dist.zip

RUN git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout v1.10.1

WORKDIR /tensorflow

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python3
ENV TF_NEED_CUDA 0
ENV TF_BUILD_ENABLE_XLA 0

RUN tensorflow/tools/ci_build/builds/configured CPU \
    bazel build -c opt --copt=-march=native --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        tensorflow/tools/pip_package:build_pip_package 

RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip3 install --upgrade -I setuptools && \ 
    pip3 --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl && \
    rm -rf /tmp/pip && \
    rm -rf /root/.cache
