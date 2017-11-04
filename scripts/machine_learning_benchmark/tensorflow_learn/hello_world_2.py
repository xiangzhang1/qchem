#!/usr/bin/python

'''
Getting started with Regression.
'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


import numpy as np
import tensorflow as tf
from functools import partial

n_inputs = 2
n_hidden1 = 4
n_hidden2 = 4
n_outputs = 1

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
training = tf.placeholder_with_default(False, shape=(), name='training')

batch_norm_layer = partial(tf.layers.batch_normalization,
                      training=training, momentum=0.9)

hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1')
bn1 = batch_norm_layer(hidden1)
bn1_act = tf.nn.elu(bn1)
hidden2 = tf.layers.dense(bn1_act, n_hidden2, name='hidden2')
bn2 = batch_norm_layer(hidden2)
bn2_act = tf.nn.elu(bn2)
output = tf.layers.dense(bn2_act, 1, name='output')

loss = tf.reduce_mean(tf.square(tf.subtract(output, y)))
optimizer = tf.train.GradientDescentOptimizer(0.01)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# execution
n_epochs = 1000
data = np.float_([
    [0, 1],
    [1, 2],
    [3, 4],
    [2, 5],
    [1, 6],
    [4, 3]
])
label = np.float_([
    [1],
    [2],
    [3],
    [4],
    [9],
    [5]
])


with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        sess.run(training_op, feed_dict={X: data, y: label})
        if epoch % 20 == 0: print epoch, loss.eval(feed_dict={X: data, y: label})
        # output.eval(feed_dict={X: data, y: label}), y.eval(feed_dict={X: data, y: label}),
    print 'Result: %s' %(output.eval(feed_dict={X: data, y: label}))
