# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tqdm

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

# tf Graph input
X = tf.placeholder("float", [None, n_input])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_perceptron(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# tf Graph input
logits = multilayer_perceptron(X)
Y = tf.nn.softmax(logits)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in tqdm.tqdm(range(mnist.test.num_examples)):
    batch_xs, batch_ys = mnist.test.next_batch(1)
    ret = sess.run(Y, feed_dict={X: batch_xs})
