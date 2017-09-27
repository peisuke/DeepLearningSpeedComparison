#!/usr/bin/env python

from __future__ import print_function

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import numpy as np
import tqdm
import time

class ConvBN(chainer.Chain):
    def __init__(self, inp, oup, stride):
        super(ConvBN, self).__init__()
        with self.init_scope():
            self.conv=L.Convolution2D(inp, oup, 3, stride=stride, pad=1, nobias=True)
            self.bn=L.BatchNormalization(oup)

    def __call__(self, x):
        h = F.relu(self.bn(self.conv(x)))
        return h

class ConvDW(chainer.Chain):
    def __init__(self, inp, oup, stride):
        super(ConvDW, self).__init__()
        with self.init_scope():
            self.conv_dw=L.DepthwiseConvolution2D(inp, 1, 3, stride=stride, pad=1, nobias=True)
            self.bn_dw=L.BatchNormalization(inp)
            self.conv_sep=L.Convolution2D(inp, oup, 1, stride=1, pad=0, nobias=True)
            self.bn_sep=L.BatchNormalization(oup)

    def __call__(self, x):
        h = F.relu(self.bn_dw(self.conv_dw(x)))
        h = F.relu(self.bn_sep(self.conv_sep(h)))
        return h
            
# Network definition
class MobileNet(chainer.Chain):
    def __init__(self):
        super(MobileNet, self).__init__()
        with self.init_scope():
            self.conv_bn = ConvBN(3, 32, 2)
            self.conv_ds_2 = ConvDW(32, 64, 1)
            self.conv_ds_3 = ConvDW(64, 128, 2)
            self.conv_ds_4 = ConvDW(128, 128, 1)
            self.conv_ds_5 = ConvDW(128, 256, 2)
            self.conv_ds_6 = ConvDW(256, 256, 1)
            self.conv_ds_7 = ConvDW(256, 512, 2)

            self.conv_ds_8 = ConvDW(512, 512, 1)
            self.conv_ds_9 = ConvDW(512, 512, 1)
            self.conv_ds_10 = ConvDW(512, 512, 1)
            self.conv_ds_11 = ConvDW(512, 512, 1)
            self.conv_ds_12 = ConvDW(512, 512, 1)

            self.conv_ds_13 = ConvDW(512, 1024, 2)
            self.conv_ds_14 = ConvDW(1024, 1024, 1)
            
    def __call__(self, x):
        x = self.conv_bn(x)
        x = self.conv_ds_2(x)
        x = self.conv_ds_3(x)
        x = self.conv_ds_4(x)
        x = self.conv_ds_5(x)
        x = self.conv_ds_6(x)
        x = self.conv_ds_7(x)
        x = self.conv_ds_8(x)
        x = self.conv_ds_9(x)
        x = self.conv_ds_10(x)
        x = self.conv_ds_11(x)
        x = self.conv_ds_12(x)
        x = self.conv_ds_13(x)
        x = self.conv_ds_14(x)
        x = F.average_pooling_2d(x, 7, stride=1)
        return F.softmax(x)

model = MobileNet()

nb_itr = 20
timings = []
for i in tqdm.tqdm(range(nb_itr)):
    data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    start_time = time.time()
    ret = F.softmax(model(chainer.Variable(data)))
    timings.append(time.time() - start_time)
print('%10s : %f (sd %f)'% ('mxnet-vgg-16', np.array(timings).mean(), np.array(timings).std()))
