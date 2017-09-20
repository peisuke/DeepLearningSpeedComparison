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
    h = F.relu(PF.affine(image, 1000, name='l1'), inplace=True)
    h = F.relu(PF.affine(h, 1000, name='l2'), inplace=True)
    h = PF.affine(h, 10, name='l3')
    return h

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
