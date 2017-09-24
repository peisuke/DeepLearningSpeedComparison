import numpy as np
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe
import tqdm

from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew

core.GlobalInit(['caffe2', '--caffe2_log_level=0'])

def AddLeNetModel(model, data):
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax

arg_scope = {"order": "NCHW"}
model = model_helper.ModelHelper(name="vgg", arg_scope=arg_scope, init_params=True)

softmax = AddLeNetModel(model, "data")

workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net, overwrite=True)

data = np.zeros([1, 3, 224, 224], np.float32)
workspace.FeedBlob("data", data)
workspace.RunNet(model.net.Proto().name)
