import numpy as np
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe
import tqdm

from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=0'])

# This section preps your image and test set in a lmdb database
def DownloadResource(url, path):
    '''Downloads resources from s3 by url and unzips them to the provided path'''
    #import requests, zipfile, StringIO
    import requests, zipfile
    from io import BytesIO
    print("Downloading... {} to {}".format(url, path))
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(BytesIO(r.content))
    z.extractall(path)
    print("Completed download and extraction.")
    
    
current_folder = './'
data_folder = os.path.join(current_folder, 'tutorial_data', 'mnist')
root_folder = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')
db_missing = False

if not os.path.exists(data_folder):
    os.makedirs(data_folder)   
    print("Your data folder was not found!! This was generated: {}".format(data_folder))

# Look for existing database: lmdb
if os.path.exists(os.path.join(data_folder,"mnist-train-nchw-lmdb")):
    print("lmdb train db found!")
else:
    db_missing = True
    
if os.path.exists(os.path.join(data_folder,"mnist-test-nchw-lmdb")):
    print("lmdb test db found!")
else:
    db_missing = True

# attempt the download of the db if either was missing
if db_missing:
    print("one or both of the MNIST lmbd dbs not found!!")
    db_url = "http://download.caffe2.ai/databases/mnist-lmdb.zip"
    try:
        DownloadResource(db_url, data_folder)
    except Exception as ex:
        print("Failed to download dataset. Please download it manually from {}".format(db_url))
        print("Unzip it and place the two database folders here: {}".format(data_folder))
        raise ex

if os.path.exists(root_folder):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree(root_folder)
    
os.makedirs(root_folder)
workspace.ResetWorkspace(root_folder)

print("training data folder:" + data_folder)
print("workspace root folder:" + root_folder)

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
