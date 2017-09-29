# Deep Learning frameworks comparison on CPU

This repository is test code for comparison of several deep learning frameworks. 
The target models are VGG-16 and MobileNet. All sample code have docker file, and
the test environment is easy to set up. Currently, the parameters of each network 
are randomly generated. I have not confirm the result is correct yet. 
I will implement weight coping as soon as possible.

## About

In this repository, the compared frameworks are as below.
- [Caffe](http://caffe.berkeleyvision.org/) : [repo](https://github.com/BVLC/caffe)
- [Caffe2](https://caffe2.ai/) : [repo](https://github.com/caffe2/caffe2)
- [Chainer](https://chainer.org/) : [repo](https://github.com/chainer/chainer)
- [Mxnet](https://mxnet.incubator.apache.org/) : [repo](https://github.com/apache/incubator-mxnet)
- [Tensorflow](https://www.tensorflow.org/) : [repo](https://github.com/tensorflow/tensorflow)
- [Pytorch](http://pytorch.org/) : [repo](https://github.com/pytorch/pytorch)
- [NNabla](https://nnabla.org/) : [repo](https://github.com/sony/nnabla)
- [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/) (Not yet) : [repo](https://github.com/Microsoft/CNTK/)
- [Theano](http://deeplearning.net/software/theano/) (Not yet) : [repo](https://github.com/Theano/Theano)

I prepared various setup condition about the frameworks, 
e.g. with/without MKL, pip or build.

## How to

Download Dockerfile and run it.

```
$ docker build -t {NAME} .
$ docker run -it --rm {NAME}
```

In the created docker containor, clone the repository and run test code.

```
# git clone https://github.com/peisuke/dl_samples.git
# cd dl_samples/{FRAMEWORK}/vgg16
# python3 (or python) predict.py
```

## Current results

__Currently, the results are not reliable. As soon as possible, I will check my code.__

```
caffe(openblas, 1.0)
caffe-vgg-16 : 13.900894 (sd 0.416803)
caffe-mobilenet : 0.121934 (sd 0.007861)

caffe(mkl, 1.0)
caffe-vgg-16 : 3.005638 (sd 0.129965)
caffe-mobilenet: 0.044592 (sd 0.010633)

caffe2(1.0)
caffe2-vgg-16 : 1.351302 (sd 0.053903)
caffe2-mobilenet : 0.069122 (sd 0.003914)

caffe2(mkl, 1.0)
caffe2-vgg-16 : 0.526263 (sd 0.026561)
caffe2-mobilenet : 0.041188 (sd 0.007531)

mxnet(0.11)
mxnet-vgg-16 : 0.896940 (sd 0.258074)
mxnet-mobilenet : 0.209141 (sd 0.060472)

mxnet(mkl)
mxnet-vgg-16 : 0.176063 (sd 0.239229)
mxnet-mobilenet : 0.022441 (sd 0.018798)

pytorch
pytorch-vgg-16 : 0.477001 (sd 0.011902)
pytorch-mobilenet : 0.094431 (sd 0.008181)

nnabla
nnabla-vgg-16 : 1.472355 (sd 0.040928)
nnabla-mobilenet : 3.984539 (sd 0.018452)

tensorflow(pip, r1.3)
tensorflow-vgg-16 : 0.275986 (sd 0.009202)
tensorflow-mobilenet : 0.029405 (sd 0.004876)

tensorflow(opt, r1.3)
tensorflow-vgg-16 : 0.144360 (sd 0.009217)
tensorflow-mobilenet : 0.022406 (sd 0.007655)

tensorflow(opt, XLA, r1.3)
tensorflow-vgg-16 : 0.151689 (sd 0.006856)
tensorflow-mobilenet : 0.022838 (sd 0.007777)

tensorflow(mkl, r1.0)
tensorflow-vgg-16 : 0.163384 (sd 0.011794)
tensorflow-mobilenet : 0.034751 (sd 0.011750)

chainer(2.0)
chainer-vgg-16 : 0.497946 (sd 0.024975)
chainer-mobilenet : 0.120230 (sd 0.013276)

chainer(2.1, numpy with mkl)
chainer-vgg-16 : 0.329744 (sd 0.013079)
chainer-vgg-16 : 0.078193 (sd 0.017298)
```
