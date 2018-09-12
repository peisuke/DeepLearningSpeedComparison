# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import tqdm
import time

_WARMUP_NUM_LOOPS = 30

def vgg(x):
    conv1_1 = tf.layers.conv2d(x, 64, 3, padding='same', activation=tf.nn.relu)
    conv1_2 = tf.layers.conv2d(conv1_1, 64, 3, padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2)
    
    conv2_1 = tf.layers.conv2d(pool1, 128, 3, padding='same', activation=tf.nn.relu)
    conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)
    
    conv3_1 = tf.layers.conv2d(pool2, 256, 3, padding='same', activation=tf.nn.relu)
    conv3_2 = tf.layers.conv2d(conv3_1, 256, 3, padding='same', activation=tf.nn.relu)
    conv3_3 = tf.layers.conv2d(conv3_2, 256, 3, padding='same', activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv3_3, 2, 2)
   
    conv4_1 = tf.layers.conv2d(pool3, 512, 3, padding='same', activation=tf.nn.relu)
    conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, padding='same', activation=tf.nn.relu)
    conv4_3 = tf.layers.conv2d(conv4_2, 512, 3, padding='same', activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(conv4_3, 2, 2)
 
    conv5_1 = tf.layers.conv2d(pool4, 512, 3, padding='same', activation=tf.nn.relu)
    conv5_2 = tf.layers.conv2d(conv5_1, 512, 3, padding='same', activation=tf.nn.relu)
    conv5_3 = tf.layers.conv2d(conv5_2, 512, 3, padding='same', activation=tf.nn.relu)
    pool5 = tf.layers.max_pooling2d(conv5_3, 2, 2)
    flat5 = tf.contrib.layers.flatten(pool5)
    
    d1 = tf.layers.dense(flat5, 4096)
    d2 = tf.layers.dense(d1, 4096)
    out = tf.layers.dense(d2, 1000)
    return tf.nn.softmax(out)

# tf Graph input
X = tf.placeholder("float", [None, 224, 224, 3])
Y = vgg(X)

init = tf.initialize_all_variables()

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

sess = tf.Session(config=config)
sess.run(init)

for i in range(_WARMUP_NUM_LOOPS):
    batch_xs = np.random.randn(1, 224, 224, 3).astype(np.float32)
    _ = sess.run(Y, feed_dict={X: batch_xs})

nb_itr = 20
timings = []
for i in tqdm.tqdm(range(nb_itr)):
    batch_xs = np.random.randn(1, 224, 224, 3).astype(np.float32)
    start_time = time.time()
    ret = sess.run(Y, feed_dict={X: batch_xs})
    timings.append(time.time() - start_time)
print('%10s : %f (sd %f)'% ('tensorflow-vgg-16', np.array(timings).mean(), np.array(timings).std()))
