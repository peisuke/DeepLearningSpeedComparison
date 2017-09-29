FROM nvidia/cuda:8.0-cudnn5-devel
 
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgoogle-glog-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgflags-dev \
    python3-dev \
    python3-pip \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopenmpi-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    python3-pydot \
    python3-setuptools \
    python3-wheel \
    python3-tk \
    libgtk2.0-0 \
    libsm6 && \ 
    rm -rf /var/lib/apt/lists/*


RUN pip3 install \
    numpy \
    protobuf \
    flask \
    graphviz \
    hypothesis \
    matplotlib \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    tornado \
    future

RUN git clone https://github.com/opencv/opencv.git && \
    mkdir opencv/build && \
    cd opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D WITH_FFMPEG=OFF \
    -D WITH_CUDA=OFF \
    -D WITH_GTK=ON \
    -D WITH_VTK=OFF \
    -D INSTALL_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    .. && make all -j4 && make install && rm -rf opencv

RUN git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2 && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF .. && \
    make -j"$(nproc)" && make install

ENV PYTHONPATH $PYTHONPATH:/caffe2/build
