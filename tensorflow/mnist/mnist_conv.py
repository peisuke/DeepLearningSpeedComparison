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

def multilayer_perceptron(x):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
    fc1 = tf.contrib.layers.flatten(conv2)
    fc1 = tf.layers.dense(fc1, 1024)
    out = tf.layers.dense(fc1, n_classes)
    return out

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
