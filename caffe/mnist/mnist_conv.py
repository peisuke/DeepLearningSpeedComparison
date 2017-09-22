import os
import PIL.Image
from StringIO import StringIO
import lmdb
import numpy as np
import sys

caffe_root = os.getenv('CAFFE_ROOT')
sys.path.insert(0, caffe_root + 'python')

import caffe

#Read lmdb dir and creat an array of the images in there
def readLMDB(lmdbDir):
    cursor = lmdb.open(lmdbDir, readonly=True).begin().cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    for key, value in cursor:
        datum.ParseFromString(value)
        s = StringIO()
        s.write(datum.data)
        s.seek(0)
        
        yield np.array(PIL.Image.open(s)), datum.label

for image, label in readLMDB(lmdbDir):
    pass
