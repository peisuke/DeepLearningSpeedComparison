import numpy as np
import os
import gzip
import struct
import time
import tqdm
from collections import namedtuple
import mxnet as mx

def conv_bn(inputs, oup, stride, name):
    conv = mx.symbol.Convolution(name=name, data=inputs, num_filter=oup, pad=(1, 1), kernel=(3, 3), stride=(stride, stride), no_bias=True)
    conv_bn = mx.symbol.BatchNorm(name=name+'_bn', data=conv, fix_gamma=False, eps=0.000100)
    out = mx.symbol.Activation(name=name+'relu', data=conv_bn, act_type='relu')
    return out 

def conv_dw(inputs, inp, oup, stride, name):
    conv_dw = mx.symbol.Convolution(name=name+'_dw', data=inputs, num_filter=inp, pad=(1, 1), kernel=(3, 3), stride=(stride, stride), no_bias=True, num_group=inp)
    conv_dw_bn = mx.symbol.BatchNorm(name=name+'dw_bn', data=conv_dw, fix_gamma=False, eps=0.000100)
    out1 = mx.symbol.Activation(name=name+'_dw', data=conv_dw_bn, act_type='relu')

    conv_sep = mx.symbol.Convolution(name=name+'_sep', data=out1, num_filter=oup, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    conv_sep_bn = mx.symbol.BatchNorm(name=name+'_sep_bn', data=conv_sep, fix_gamma=False, eps=0.000100)
    out2 = mx.symbol.Activation(name=name+'_sep', data=conv_sep_bn, act_type='relu')
    return out2

def create_network():
    data = mx.sym.Variable('data')
    net = conv_bn(data, 32, stride=2, name='conv_bn')
    net = conv_dw(net, 32, 64, stride=1, name='conv_ds_2')
    net = conv_dw(net, 64, 128, stride=2, name='conv_ds_3')
    net = conv_dw(net, 128, 128, stride=1, name='conv_ds_4')
    net = conv_dw(net, 128, 256, stride=2, name='conv_ds_5')
    net = conv_dw(net, 256, 256, stride=1, name='conv_ds_6')
    net = conv_dw(net, 256, 512, stride=2, name='conv_ds_7')

    net = conv_dw(net, 512, 512, stride=1, name='conv_ds_8')
    net = conv_dw(net, 512, 512, stride=1, name='conv_ds_9')
    net = conv_dw(net, 512, 512, stride=1, name='conv_ds_10')
    net = conv_dw(net, 512, 512, stride=1, name='conv_ds_11')
    net = conv_dw(net, 512, 512, stride=1, name='conv_ds_12')

    net = conv_dw(net, 512, 1024, stride=2, name='conv_ds_13')
    net = conv_dw(net, 1024, 1024, stride=1, name='conv_ds_14')
    net = mx.symbol.Pooling(data=net, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

    return mx.sym.softmax(net)

mlp = create_network()
mod = mx.mod.Module(symbol=mlp, context=mx.cpu(), label_names=None)
mod.bind(data_shapes=[('data', (1, 3, 224, 224))], for_training=False)
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
Batch = namedtuple('Batch', ['data'])

nb_itr = 20
timings = []
for i in tqdm.tqdm(range(nb_itr)):
    data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    start_time = time.time()
    batch = Batch([mx.nd.array(data)])
    mod.forward(batch)
    prob = mod.get_outputs()[0].asnumpy()
    timings.append(time.time() - start_time)
print('%10s : %f (sd %f)'% ('mxnet-mobilenet', np.array(timings).mean(), np.array(timings).std()))
