import numpy as np
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe

from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew

core.GlobalInit(['caffe2', '--caffe2_log_level=0'])

def AddLeNetModel(model, data):
    conv1_1 = brew.conv(model, data, 'conv1_1', dim_in=3, dim_out=64, kernel=3, pad=1)
    conv1_1 = brew.relu(model, conv1_1, conv1_1)
    conv1_2 = brew.conv(model, conv1_1, 'conv1_2', dim_in=64, dim_out=64, kernel=3, pad=1)
    conv1_2 = brew.relu(model, conv1_2, conv1_2)
    pool1 = brew.max_pool(model, conv1_2, 'pool1', kernel=2, stride=2)
    
    conv2_1 = brew.conv(model, pool1, 'conv2_1', dim_in=64, dim_out=128, kernel=3, pad=1)
    conv2_1 = brew.relu(model, conv2_1, conv2_1)
    conv2_2 = brew.conv(model, conv2_1, 'conv2_2', dim_in=128, dim_out=128, kernel=3, pad=1)
    conv2_2 = brew.relu(model, conv2_2, conv2_2)
    pool2 = brew.max_pool(model, conv2_2, 'pool2', kernel=2, stride=2)
    
    conv3_1 = brew.conv(model, pool2, 'conv3_1', dim_in=128, dim_out=256, kernel=3, pad=1)
    conv3_1 = brew.relu(model, conv3_1, conv3_1)
    conv3_2 = brew.conv(model, conv3_1, 'conv3_2', dim_in=256, dim_out=256, kernel=3, pad=1)
    conv3_2 = brew.relu(model, conv3_2, conv3_2)
    conv3_3 = brew.conv(model, conv3_2, 'conv3_3', dim_in=256, dim_out=256, kernel=3, pad=1)
    conv3_3 = brew.relu(model, conv3_3, conv3_3)
    pool3 = brew.max_pool(model, conv3_3, 'pool3', kernel=2, stride=2)
   
    conv4_1 = brew.conv(model, pool3, 'conv4_1', dim_in=256, dim_out=512, kernel=3, pad=1)
    conv4_1 = brew.relu(model, conv4_1, conv4_1)
    conv4_2 = brew.conv(model, conv4_1, 'conv4_2', dim_in=512, dim_out=512, kernel=3, pad=1)
    conv4_2 = brew.relu(model, conv4_2, conv4_2)
    conv4_3 = brew.conv(model, conv4_2, 'conv4_3', dim_in=512, dim_out=512, kernel=3, pad=1)
    conv4_3 = brew.relu(model, conv4_3, conv4_3)
    pool4 = brew.max_pool(model, conv4_3, 'pool4', kernel=2, stride=2)
 
    conv5_1 = brew.conv(model, pool4, 'conv5_1', dim_in=512, dim_out=512, kernel=3, pad=1)
    conv5_1 = brew.relu(model, conv5_1, conv5_1)
    conv5_2 = brew.conv(model, conv5_1, 'conv5_2', dim_in=512, dim_out=512, kernel=3, pad=1)
    conv5_2 = brew.relu(model, conv5_2, conv5_2)
    conv5_3 = brew.conv(model, conv5_2, 'conv5_3', dim_in=512, dim_out=512, kernel=3, pad=1)
    conv5_3 = brew.relu(model, conv5_3, conv5_3)
    pool5 = brew.max_pool(model, conv5_3, 'pool5', kernel=2, stride=2)
 
    fc6 = brew.fc(model, pool5, 'fc6', dim_in=25088, dim_out=4096)
    fc6 = brew.relu(model, fc6, fc6)
    fc7 = brew.fc(model, fc6, 'fc7', dim_in=4096, dim_out=4096)
    fc7 = brew.relu(model, fc7, fc7)
    pred = brew.fc(model, fc7, 'pred', 4096, 1000)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax

model = model_helper.ModelHelper(name="vgg", init_params=True)

softmax = AddLeNetModel(model, "data")

workspace.RunNetOnce(model.param_init_net)

data = np.zeros([1, 3, 224, 224], np.float32)
workspace.FeedBlob("data", data)

workspace.CreateNet(model.net)
workspace.RunNet(model.net.Proto().name)
ref_out = workspace.FetchBlob("softmax")
