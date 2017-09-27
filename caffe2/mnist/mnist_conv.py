import numpy as np
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe
import tqdm

from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew
from mnist_data import DownloadMNIST

def AddInput(model, batch_size, db, db_type):
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    data = model.Scale(data, data, scale=float(1./256))
    data = model.StopGradient(data, data)
    return data, label

def AddLeNetModel(model, data):
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=32, kernel=3, pad=1)
    conv1 = brew.relu(model, conv1, conv1)
    conv2 = brew.conv(model, conv1, 'conv2', dim_in=32, dim_out=128, kernel=3, pad=1)
    conv2 = brew.relu(model, conv2, conv2)
    conv3 = brew.conv(model, conv2, 'conv3', dim_in=128, dim_out=256, kernel=3, pad=1)
    conv3 = brew.relu(model, conv3, conv3)
    fc3 = brew.fc(model, conv3, 'fc3', dim_in=256 * 28 * 28, dim_out=512)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 512, 10)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax

core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
root_folder, data_folder = DownloadMNIST()
workspace.ResetWorkspace(root_folder)

arg_scope = {"order": "NCHW"}
test_model = model_helper.ModelHelper(
    name="mnist_test", arg_scope=arg_scope, init_params=True)
data, label = AddInput(
    test_model, batch_size=1,
    db=os.path.join(data_folder, 'mnist-test-nchw-lmdb'),
    db_type='lmdb')

softmax = AddLeNetModel(test_model, data)

# run a test pass on the test net
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite=True)
test_accuracy = np.zeros(10000)
for i in tqdm.tqdm(range(10000)):
    workspace.RunNet(test_model.net.Proto().name)
