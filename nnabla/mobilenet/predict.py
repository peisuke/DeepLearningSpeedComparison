import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
from nnabla.contrib.context import extension_context

def conv_bn(inputs, oup, stride, name):
    h = PF.convolution(inputs, oup, (3, 3), pad=(1, 1), stride=(stride, stride), with_bias=False, name=name) 
    h = PF.batch_normalization(h, name=name+'_bn')
    h = F.relu(h)
    return h

def conv_dw(inputs, inp, oup, stride, name):
    h = PF.convolution(inputs, inp, (3, 3), pad=(1, 1), stride=(stride, stride), with_bias=None, group=inp, name=name+'_dw') 
    h = PF.batch_normalization(h, name=name+'_dw_bn')
    h = F.relu(h)

    h = PF.convolution(inputs, oup, (1, 1), pad=(0, 0), stride=(1, 1), with_bias=None, name=name+'_sep') 
    h = PF.batch_normalization(h, name=name+'_sep_bn')
    h = F.relu(h)

    return h

def mobilenet(image, test=False):
    net = conv_bn(image, 32, stride=2, name='conv_bn')
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
    h = F.average_pooling(net, (7, 7))

    return h

# Get context.
ctx = extension_context('cpu', device_id=0)
nn.set_default_context(ctx)

# Create input variables.
vimage = nn.Variable([1, 3, 224, 224])
vpred = mobilenet(vimage, test=True)

vimage.d = np.zeros([1, 3, 224, 224], np.float32)
vpred.forward(clear_buffer=True)
