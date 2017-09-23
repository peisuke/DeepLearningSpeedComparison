import numpy as np
import os
import gzip
import struct
import tqdm
from collections import namedtuple
import mxnet as mx

def create_network():
    data = mx.sym.Variable('data')
    h = mx.sym.Convolution(data, kernel=(3, 3), pad=(1, 1), num_filter=64, name = "conv1-1")
    h = mx.sym.Activation(h, name='relu1-1', act_type="relu")
    h = mx.sym.Convolution(h, kernel=(3, 3), pad=(1, 1), num_filter=64, name = "conv1-2")
    h = mx.sym.Activation(h, name='relu1-2', act_type="relu")
    h = mx.sym.Pooling(h, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    
    h = mx.sym.Convolution(h, kernel=(3, 3), pad=(1, 1), num_filter=128, name = "conv2-1")
    h = mx.sym.Activation(h, name='relu2-1', act_type="relu")
    h = mx.sym.Convolution(h, kernel=(3, 3), pad=(1, 1), num_filter=128, name = "conv2-2")
    h = mx.sym.Activation(h, name='relu2-2', act_type="relu")
    h = mx.sym.Pooling(h, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    
    h = mx.sym.Convolution(h, kernel=(3, 3), pad=(1, 1), num_filter=256, name = "conv3-1")
    h = mx.sym.Activation(h, name='relu3-1', act_type="relu")
    h = mx.sym.Convolution(h, kernel=(3, 3), pad=(1, 1), num_filter=256, name = "conv3-2")
    h = mx.sym.Activation(h, name='relu3-2', act_type="relu")
    h = mx.sym.Convolution(h, kernel=(3, 3), pad=(1, 1), num_filter=256, name = "conv3-3")
    h = mx.sym.Activation(h, name='relu3-3', act_type="relu")
    h = mx.sym.Pooling(h, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    
    h = mx.sym.Convolution(h, kernel=(3, 3), pad=(1, 1), num_filter=512, name = "conv4-1")
    h = mx.sym.Activation(h, name='relu4-1', act_type="relu")
    h = mx.sym.Convolution(h, kernel=(3, 3), pad=(1, 1), num_filter=512, name = "conv4-2")
    h = mx.sym.Activation(h, name='relu4-2', act_type="relu")
    h = mx.sym.Convolution(h, kernel=(3, 3), pad=(1, 1), num_filter=512, name = "conv4-3")
    h = mx.sym.Activation(h, name='relu4-3', act_type="relu")
    h = mx.sym.Pooling(h, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    
    h = mx.sym.Convolution(h, kernel=(3, 3), pad=(1, 1), num_filter=512, name = "conv5-1")
    h = mx.sym.Activation(h, name='relu5-1', act_type="relu")
    h = mx.sym.Convolution(h, kernel=(3, 3), pad=(1, 1), num_filter=512, name = "conv5-2")
    h = mx.sym.Activation(h, name='relu5-2', act_type="relu")
    h = mx.sym.Convolution(h, kernel=(3, 3), pad=(1, 1), num_filter=512, name = "conv5-3")
    h = mx.sym.Activation(h, name='relu5-3', act_type="relu")
    h = mx.sym.Pooling(h, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    
    h = mx.sym.Flatten(h)
    
    h = mx.sym.FullyConnected(h, name='fc6', num_hidden = 4096)
    h = mx.sym.Activation(h, name='relu6', act_type="relu")
    h = mx.sym.FullyConnected(h, name='fc7', num_hidden = 4096)
    h = mx.sym.Activation(h, name='relu7', act_type="relu")
    h = mx.sym.FullyConnected(h, name='fc8', num_hidden=1000)
    
    return h

mlp = create_network()
mod = mx.mod.Module(symbol=mlp, context=mx.cpu(), label_names=None)
mod.bind(data_shapes=[('data', (1, 3, 224, 224))], for_training=False)
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

Batch = namedtuple('Batch', ['data'])
data = np.zeros([1, 3, 224, 224], np.float32)
batch = Batch([mx.nd.array(data)])
prob = mod.forward(batch)
