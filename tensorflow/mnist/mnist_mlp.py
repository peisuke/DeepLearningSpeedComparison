# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tqdm

weights = {
    'h1': tf.Variable(tf.random_normal([784, 1000])),
    'h2': tf.Variable(tf.random_normal([1000, 1000])),
    'out': tf.Variable(tf.random_normal([1000, 10]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([1000])),
    'b2': tf.Variable(tf.random_normal([1000])),
    'out': tf.Variable(tf.random_normal([10]))
}

def multilayer_perceptron(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return tf.nn.softmax(out_layer)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder("float", [None, 784])
Y = multilayer_perceptron(X)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in tqdm.tqdm(range(mnist.test.num_examples)):
    batch_xs, batch_ys = mnist.test.next_batch(1)
    ret = sess.run(Y, feed_dict={X: batch_xs})
