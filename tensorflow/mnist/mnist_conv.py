# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tqdm

def net(x):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(x, 32, 3, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, 128, 3, padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, 256, 3, padding='same', activation=tf.nn.relu)
    fc1 = tf.contrib.layers.flatten(conv3)
    fc1 = tf.layers.dense(fc1, 512)
    out = tf.layers.dense(fc1, 10)
    return tf.nn.softmax(out)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder("float", [None, 784])
Y = net(X)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in tqdm.tqdm(range(mnist.test.num_examples)):
    batch_xs, batch_ys = mnist.test.next_batch(1)
    ret = sess.run(Y, feed_dict={X: batch_xs})
