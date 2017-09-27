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
    h = mx.sym.Convolution(data, kernel=(3, 3), pad=(1, 1), num_filter=32, name = "conv1")
    h = mx.sym.Activation(h, name='relu1', act_type="relu")
    h = mx.sym.Convolution(data, kernel=(3, 3), pad=(1, 1), num_filter=128, name = "conv2")
    h = mx.sym.Activation(h, name='relu2', act_type="relu")
    h = mx.sym.Convolution(data, kernel=(3, 3), pad=(1, 1), num_filter=256, name = "conv3")
    h = mx.sym.Activation(h, name='relu3', act_type="relu")
    h = mx.sym.Flatten(h)
    h = mx.sym.FullyConnected(h, name='fc3', num_hidden = 512)
    h = mx.sym.Activation(h, name='relu4', act_type="relu")
    h = mx.sym.FullyConnected(h, name='fc4', num_hidden=10)
    return mx.sym.softmax(h)
 
mlp = create_network()
mod = mx.mod.Module(symbol=mlp, context=mx.cpu(), label_names=None)
mod.bind(data_shapes=[('data', (1, 1, 28, 28))], for_training=False)
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

for img in tqdm.tqdm(val_img):
    Batch = namedtuple('Batch', ['data'])
    data = img.astype(np.float32) / 255 
    data = data[np.newaxis, np.newaxis, :, :]
    batch = Batch([mx.nd.array(data)])
    prob = mod.forward(batch)
