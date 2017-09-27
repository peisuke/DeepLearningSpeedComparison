import numpy as np
import os
import urllib.request
import gzip
import struct
import tqdm
from collections import namedtuple
import mxnet as mx
from mnist_data import read_data

path='http://yann.lecun.com/exdb/mnist/'
(train_lbl, train_img) = read_data(path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
(val_lbl, val_img) = read_data(path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')

def create_network():
    data = mx.sym.Variable('data')
    data = mx.sym.Flatten(data)
    fc1  = mx.sym.FullyConnected(data, name='fc1', num_hidden=1000)
    act1 = mx.sym.Activation(fc1, name='relu1', act_type="relu")
    fc2  = mx.sym.FullyConnected(act1, name='fc2', num_hidden = 1000)
    act2 = mx.sym.Activation(fc2, name='relu2', act_type="relu")
    fc3  = mx.sym.FullyConnected(act2, name='fc3', num_hidden=10)
    return mx.sym.softmax(fc3)
 
mlp = create_network()
mod = mx.mod.Module(symbol=mlp, context=mx.cpu(), label_names=None)
mod.bind(data_shapes=[('data', (1, 1, 28, 28))], for_training=False)
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

for img in tqdm.tqdm(val_img):
    Batch = namedtuple('Batch', ['data'])
    batch = Batch([mx.nd.array([img.astype(np.float32)/255])])
    prob = mod.forward(batch)
