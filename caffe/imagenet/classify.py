import os
import PIL.Image
from StringIO import StringIO
import lmdb
import numpy as np
import sys

caffe_root = os.getenv('CAFFE_ROOT')
sys.path.insert(0, caffe_root + 'python')
import caffe

MODEL_FILE = './deploy.prototxt'
PRETRAINED = './bvlc_googlenet.caffemodel'

caffe.set_mode_gpu()

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
               mean=np.load('./ilsvrc_2012_mean.npy').mean(1).mean(1),
               channel_swap=(2,1,0),
               raw_scale=255,
               image_dims=(224, 224))

def caffe_predict(path):
    input_image = caffe.io.load_image(path)
    prediction = net.predict([input_image])
    proba = prediction[0][prediction[0].argmax()]
    ind = prediction[0].argsort()[-5:][::-1] # top-5 predictions
    return prediction[0].argmax(), proba, ind

# load ImageNet labels
labels = np.loadtxt('synset_words.txt', str, delimiter='\t')

p, prob, _ = caffe_predict('./cat.jpg')
print(labels[p], prob)
