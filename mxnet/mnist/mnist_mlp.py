import numpy as np
import os
import urllib.request
import gzip
import struct
import tqdm
from collections import namedtuple
import mxnet as mx

def download_data(url, force_download=True):
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        urllib.request.urlretrieve(url, fname)
    return fname
 
def read_data(label_url, image_url):
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)
 
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
    return fc3 
 
mlp = create_network()
mod = mx.mod.Module(symbol=mlp, context=mx.cpu(), label_names=None)
mod.bind(data_shapes=[('data', (1, 1, 28, 28))], for_training=False)
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

for img in tqdm.tqdm(val_img):
    Batch = namedtuple('Batch', ['data'])
    batch = Batch([mx.nd.array([img.astype(np.float32)/255])])
    prob = mod.forward(batch)
