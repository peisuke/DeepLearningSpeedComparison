import os
import PIL.Image
from StringIO import StringIO
import numpy as np
import sys
import tqdm
import time

caffe_root = os.getenv('CAFFE_ROOT')
sys.path.insert(0, caffe_root + 'python')
import caffe

MODEL_FILE = './mobilenet_deploy.prototxt'
PRETRAINED = './mobilenet.caffemodel'

caffe.set_mode_cpu()

cnn = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

nb_itr = 20
timings = []
for i in tqdm.tqdm(range(nb_itr)):
    cnn.blobs['data'].data[...] = np.random.randn(1, 3, 224, 224).astype(np.float32) 
    start_time = time.time()
    output = cnn.forward()
    timings.append(time.time() - start_time)
print('%10s : %f (sd %f)'% ('caffe-vgg-16', np.array(timings).mean(), np.array(timings).std()))
