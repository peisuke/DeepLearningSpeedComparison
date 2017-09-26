# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def conv_bn(inputs, oup, stride, sc):
    conv = slim.convolution2d(inputs,
                              oup,
                              kernel_size=[3, 3],
                              stride = stride,
                              padding='SAME',
                              scope=sc+'/conv')
    bn = slim.batch_norm(conv, scope=sc+'/bn')
    return bn

def conv_dw(inputs, oup, stride, sc):
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[3, 3],
                                                  scope=sc+'/dw_conv')

    bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_bn')
    pointwise_conv = slim.convolution2d(bn,
                                        oup,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pw_conv')
    bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_bn')
    return bn

def mobilenet(x):
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        activation_fn=None):
        with slim.arg_scope([slim.batch_norm],
                            is_training=False,
                            activation_fn=tf.nn.relu):
            net = conv_bn(x, 32, stride=2, sc='conv_bn')
            net = conv_dw(net, 64, stride=1, sc='conv_ds_2')
            net = conv_dw(net, 128, stride=2, sc='conv_ds_3')
            net = conv_dw(net, 128, stride=1, sc='conv_ds_4')
            net = conv_dw(net, 256, stride=2, sc='conv_ds_5')
            net = conv_dw(net, 256, stride=1, sc='conv_ds_6')
            net = conv_dw(net, 512, stride=2, sc='conv_ds_7')

            net = conv_dw(net, 512, stride=1, sc='conv_ds_8')
            net = conv_dw(net, 512, stride=1, sc='conv_ds_9')
            net = conv_dw(net, 512, stride=1, sc='conv_ds_10')
            net = conv_dw(net, 512, stride=1, sc='conv_ds_11')
            net = conv_dw(net, 512, stride=1, sc='conv_ds_12')

            net = conv_dw(net, 1024, stride=2, sc='conv_ds_13')
            net = conv_dw(net, 1024, stride=1, sc='conv_ds_14')
            net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')
            return tf.nn.softmax(net)

# tf Graph input
X = tf.placeholder("float", [None, 224, 224, 3])
Y = mobilenet(X)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

batch_xs = np.zeros([1, 224, 224, 3], np.float32)
ret = sess.run(Y, feed_dict={X: batch_xs})
