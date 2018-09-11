#!/usr/bin/env python

from __future__ import print_function

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.backends.intel64 import is_ideep_available

import numpy as np
import tqdm
import time

# Network definition
class VGG(chainer.Chain):
    def __init__(self):
        super(VGG, self).__init__()
        with self.init_scope():
            self.conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1)
            self.conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1)

            self.conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1)
            self.conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1)

            self.conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1)
            self.conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1)

            self.conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1)
            self.conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1)

            self.conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1)

            self.fc6=L.Linear(25088, 4096)
            self.fc7=L.Linear(4096, 4096)
            self.fc8=L.Linear(4096, 1000)

    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)        
        
        return h

def main():
    enable_ideep = is_ideep_available()
    model = VGG()
    mode = "never"
    if enable_ideep:
        model.to_intel64()
        mode = "always"

    nb_itr = 20
    timings = []
    for i in tqdm.tqdm(range(nb_itr)):
        data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        start_time = time.time()
        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                with chainer.using_config('use_ideep', mode):
    	            ret = F.softmax(model(chainer.Variable(data)))
        print(ret.data.ravel()[0])
        timings.append(time.time() - start_time)
    print('%10s : %f (sd %f)'% ('chainer-vgg-16', np.array(timings).mean(), np.array(timings).std()))

if __name__ == '__main__':
    main()
