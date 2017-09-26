import os
import PIL.Image
from StringIO import StringIO
import numpy as np
import sys

caffe_root = os.getenv('CAFFE_ROOT')
sys.path.insert(0, caffe_root + 'python')
import caffe

MODEL_FILE = './mobilenet_deploy.prototxt'
PRETRAINED = './mobilenet.caffemodel'

caffe.set_mode_cpu()

mean = np.array([103.94,116.78,123.68], dtype=np.float32)
channel_swap = (2,1,0)
raw_scale = 255
image_dims=(224, 224)

cnn = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

in_ = cnn.inputs[0]
transformer = caffe.io.Transformer({in_: cnn.blobs[in_].data.shape})
transformer.set_transpose(in_, (2, 0, 1))
transformer.set_raw_scale(in_, raw_scale)
transformer.set_mean(in_, mean)
transformer.set_channel_swap(in_, channel_swap)

input_image = caffe.io.load_image('../vgg16/cat.jpg')
input_image = caffe.io.resize_image(input_image, image_dims)
input_image = transformer.preprocess(cnn.inputs[0], input_image)
input_image = input_image[np.newaxis, :, :, :]

cnn.blobs['data'].data[...] = input_image 
output = cnn.forward()
pred = output[cnn.outputs[0]][0]

# load ImageNet labels
labels = np.loadtxt('synset_words.txt', str, delimiter='\t')

print(labels[pred.argmax()], pred.max())
