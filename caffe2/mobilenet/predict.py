import numpy as np
import os
import tqdm
import time
import shutil
import caffe2.python.predictor.predictor_exporter as pe

from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags

core.GlobalInit(['caffe2', '--caffe2_log_level=0'])

def conv_bn(model, inputs, inp, oup, stride, name):
    h = brew.conv(model, inputs, name, dim_in=inp, dim_out=oup, kernel=3, pad=1, stride=stride, no_bias=True)
    h = brew.spatial_bn(model, h, name+'_bn', oup, is_test=True)
    h = brew.relu(model, h, h)
    return h 

def conv_dw(model, inputs, inp, oup, stride, name):
    W_dw = model.create_param(
            param_name=name + '_dw_w',
            shape=[inp, 1, 3, 3],
            initializer=initializers.update_initializer(None, None, ("XavierFill", {})),
            tags=ParameterTags.WEIGHT
        )
    h = inputs.Conv([W_dw], [name+'_dw'], kernel_h=3, kernel_w=3, stride_h=stride, stride_w=stride, 
                    pad_b=1, pad_l=1, pad_r=1, pad_t=1, order='NCHW', group=inp)
    h = brew.spatial_bn(model, h, name+'_dw_bn', inp, is_test=True)
    h = brew.relu(model, h, h)
    
    h = brew.conv(model, h, name+'_sep', dim_in=inp, dim_out=oup, kernel=1, pad=0, stride=1, no_bias=True)
    h = brew.spatial_bn(model, h, name+'_sep_bn', oup, is_test=True)
    h = brew.relu(model, h, h)
    return h

def AddLeNetModel(model, data):
    h = conv_bn(model, data, 3, 32, stride=2, name='conv_bn')
    h = conv_dw(model, h, 32, 64, stride=1, name='conv_ds_2') 
    h = conv_dw(model, h, 64, 128, stride=2, name='conv_ds_3')
    h = conv_dw(model, h, 128, 128, stride=1, name='conv_ds_4')
    h = conv_dw(model, h, 128, 256, stride=2, name='conv_ds_5')
    h = conv_dw(model, h, 256, 256, stride=1, name='conv_ds_6')
    h = conv_dw(model, h, 256, 512, stride=2, name='conv_ds_7')

    h = conv_dw(model, h, 512, 512, stride=1, name='conv_ds_8')
    h = conv_dw(model, h, 512, 512, stride=1, name='conv_ds_9')
    h = conv_dw(model, h, 512, 512, stride=1, name='conv_ds_10')
    h = conv_dw(model, h, 512, 512, stride=1, name='conv_ds_11')
    h = conv_dw(model, h, 512, 512, stride=1, name='conv_ds_12')

    h = conv_dw(model, h, 512, 1024, stride=2, name='conv_ds_13')
    h = conv_dw(model, h, 1024, 1024, stride=1, name='conv_ds_14')
    h = brew.average_pool(model, h, 'pool5', kernel=7) 
    softmax = brew.softmax(model, h, 'softmax')
    return softmax

model = model_helper.ModelHelper(name="mobilenet", init_params=True)
softmax = AddLeNetModel(model, "data")
workspace.RunNetOnce(model.param_init_net)

data = np.zeros([1, 3, 224, 224], np.float32)
workspace.FeedBlob("data", data)
workspace.CreateNet(model.net)

nb_itr = 20
timings = []
for i in tqdm.tqdm(range(nb_itr)):
    data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    start_time = time.time()
    workspace.FeedBlob("data", data)
    workspace.RunNet(model.net.Proto().name)
    ref_out = workspace.FetchBlob("softmax")
    timings.append(time.time() - start_time)
print('%10s : %f (sd %f)'% ('caffe2-mobilenet', np.array(timings).mean(), np.array(timings).std()))
