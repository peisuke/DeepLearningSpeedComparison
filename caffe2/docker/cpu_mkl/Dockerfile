FROM ubuntu:16.04
 
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

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ gfortran wget cpio && \
  cd /tmp && \
  wget -q http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/12070/l_mkl_2018.0.128.tgz && \
  tar -xzf l_mkl_2018.0.128.tgz && \
  cd l_mkl_2018.0.128 && \
  sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/g' silent.cfg && \
  sed -i 's/ARCH_SELECTED=ALL/ARCH_SELECTED=INTEL64/g' silent.cfg && \
  sed -i 's/COMPONENTS=DEFAULTS/COMPONENTS=;intel-comp-l-all-vars__noarch;intel-comp-nomcu-vars__noarch;intel-openmp__x86_64;intel-tbb-libs__x86_64;intel-mkl-common__noarch;intel-mkl-installer-license__noarch;intel-mkl-core__x86_64;intel-mkl-core-rt__x86_64;intel-mkl-doc__noarch;intel-mkl-doc-ps__noarch;intel-mkl-gnu__x86_64;intel-mkl-gnu-rt__x86_64;intel-mkl-common-ps__noarch;intel-mkl-core-ps__x86_64;intel-mkl-common-c__noarch;intel-mkl-core-c__x86_64;intel-mkl-common-c-ps__noarch;intel-mkl-tbb__x86_64;intel-mkl-tbb-rt__x86_64;intel-mkl-gnu-c__x86_64;intel-mkl-common-f__noarch;intel-mkl-core-f__x86_64;intel-mkl-gnu-f-rt__x86_64;intel-mkl-gnu-f__x86_64;intel-mkl-f95-common__noarch;intel-mkl-f__x86_64;intel-mkl-psxe__noarch;intel-psxe-common__noarch;intel-psxe-common-doc__noarch;intel-compxe-pset/g' silent.cfg && \
  ./install.sh -s silent.cfg && \
  cd .. && rm -rf * && \
  rm -rf /opt/intel/.*.log /opt/intel/compilers_and_libraries_2018.0.128/licensing && \
  echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/intel.conf && \
  ldconfig && \
  echo "source /opt/intel/mkl/bin/mklvars.sh intel64" >> /etc/bash.bashrc

RUN git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2 && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DBLAS=MKL .. && \
    make -j"$(nproc)" && make install

ENV PYTHONPATH $PYTHONPATH:/caffe2/build
