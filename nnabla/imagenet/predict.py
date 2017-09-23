import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
from nnabla.contrib.context import extension_context

def vgg(image, test=False):
    image /= 255.0
    h = F.relu(PF.convolution(image, 64, (3, 3), pad=(1, 1), stride=(1, 1), name='conv1'))
    h = F.relu(PF.convolution(h, 64,  (3, 3), pad=(1, 1), stride=(1, 1), name='conv2'))
    h = F.max_pooling(h, (2, 2))
    h = F.relu(PF.convolution(h, 128, (3, 3), pad=(1, 1), stride=(1, 1), name='conv3'))
    h = F.relu(PF.convolution(h, 128, (3, 3), pad=(1, 1), stride=(1, 1), name='conv4'))
    h = F.max_pooling(h, (2, 2))
    h = F.relu(PF.convolution(h, 256, (3, 3), pad=(1, 1), stride=(1, 1), name='conv5'))
    h = F.relu(PF.convolution(h, 256, (3, 3), pad=(1, 1), stride=(1, 1), name='conv6'))
    h = F.relu(PF.convolution(h, 256, (3, 3), pad=(1, 1), stride=(1, 1), name='conv7'))
    h = F.max_pooling(h, (2, 2))
    h = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv8'))
    h = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv9'))
    h = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv10'))
    h = F.max_pooling(h, (2, 2))
    h = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv11'))
    h = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv12'))
    h = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv13'))
    h = F.max_pooling(h, (2, 2))
    h = PF.affine(h, 4096, name='fc1')
    h = F.relu(h)
    h = PF.affine(h, 4096, name='fc2')
    h = F.relu(h)
    h = PF.affine(h, 1000, name='fc3')
    return h

# Get context.
ctx = extension_context('cpu', device_id=0)
nn.set_default_context(ctx)

# Create input variables.
vimage = nn.Variable([1, 3, 224, 224])
vpred = vgg(vimage, test=True)

vimage.d = np.zeros([1, 3, 224, 224], np.float32)
vpred.forward(clear_buffer=True)
