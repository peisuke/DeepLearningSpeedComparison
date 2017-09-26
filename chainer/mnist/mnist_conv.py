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

# Network definition
class MLP(chainer.Chain):

    def __init__(self):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.c1 = L.Convolution2D(None, 32, 3, 1, 1)  # n_in -> n_units
            self.c2 = L.Convolution2D(None, 128, 3, 1, 1)  # n_units -> n_units
            self.c3 = L.Convolution2D(None, 256, 3, 1, 1)  # n_units -> n_out
            self.l4 = L.Linear(None, 512)
            self.l5 = L.Linear(None, 10)

    def __call__(self, x):
        h1 = F.relu(self.c1(x))
        h2 = F.relu(self.c2(h1))
        h3 = F.relu(self.c3(h2))
        h4 = F.relu(self.l4(h3))
        return self.l5(h4)

# Load the MNIST dataset
_, test = chainer.datasets.get_mnist(ndim=3)

model = MLP()

for img, label in tqdm.tqdm(test):
    x_data = chainer.Variable(img[np.newaxis,:,:,:])
    ret = F.softmax(model(x_data)).data
