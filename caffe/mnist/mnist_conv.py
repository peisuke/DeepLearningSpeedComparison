import os
import PIL.Image
from StringIO import StringIO
import lmdb
import numpy as np
import sys
import tqdm

caffe_root = os.getenv('CAFFE_ROOT')
sys.path.insert(0, caffe_root + 'python')

import caffe

def readLMDB(lmdbDir):
    cursor = lmdb.open(lmdbDir, readonly=True).begin().cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    images = []
    labels = []
    for key, value in cursor:
        datum.ParseFromString(value)
        img = caffe.io.datum_to_array(datum)
        images.append(img) 
        labels.append(datum.label)
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels

lmdbDir = './examples/mnist/mnist_test_lmdb'
images, labels = readLMDB(lmdbDir)

cnn = caffe.Net('deploy.prototxt', './examples/mnist/lenet_iter_10000.caffemodel', caffe.TEST)

for img, label in tqdm.tqdm(zip(images, labels)):
    cnn.blobs['data'].data[...] = img
    output = cnn.forward()
    pred = output[cnn.outputs[0]][0]
