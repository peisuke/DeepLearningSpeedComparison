import os
import tqdm
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
from nnabla.contrib.context import extension_context

from mnist_data import data_iterator_mnist

def mlp(image, test=False):
    image /= 255.0
    c1 = F.relu(PF.convolution(image, 32, (3, 3), name='conv1'), inplace=True)
    c2 = F.relu(PF.convolution(c1, 128, (3, 3), name='conv2'), inplace=True)
    c3 = F.relu(PF.convolution(c2, 256, (3, 3), name='conv3'), inplace=True)
    c4 = F.relu(PF.affine(c3, 512, name='fc3'), inplace=True)
    c5 = PF.affine(c3, 10, name='fc4')
    return F.softmax(c5)

# Get context.
ctx = extension_context('cpu', device_id=0)
nn.set_default_context(ctx)

# Create CNN network for both training and testing.
mnist_cnn_prediction = mlp

# Create input variables.
vimage = nn.Variable([1, 1, 28, 28])
vpred = mnist_cnn_prediction(vimage, test=True)

# Initialize DataIterator for MNIST.
vdata = data_iterator_mnist(1, False)

for j in tqdm.tqdm(range(vdata.size)):
    vimage.d, _ = vdata.next()
    vpred.forward(clear_buffer=True)
