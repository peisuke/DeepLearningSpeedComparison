#!/usr/bin/env python

from __future__ import print_function

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
            self.l1 = L.Linear(784, 1000)  # n_in -> n_units
            self.l2 = L.Linear(1000, 1000)  # n_units -> n_units
            self.l3 = L.Linear(1000, 10)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

model = MLP()

_, test = chainer.datasets.get_mnist()

for img, label in tqdm.tqdm(test):
    x_data = chainer.Variable(img[np.newaxis,:])
    ret = F.softmax(model(x_data)).data
