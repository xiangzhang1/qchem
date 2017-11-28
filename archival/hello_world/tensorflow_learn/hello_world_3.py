#!/usr/bin/env python
'''
Getting started with pipelines and batches
'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import numpy as np
import tensorflow as tf

# data
X_train = np.c_[[0]*100,range(100)] # [[0,1],[0,2],...,[0,100]]
y_train = np.c_[range(100)] #[[1],[2],...,[100]]

# construction
n_inputs = 2
batch_size = 20

# X = tf.placeholder(tf.float32, shape=(100, n_inputs), name='X')
# y = tf.placeholder(tf.float32, shape=(100, 1), name='y')

# import pdb
# pdb.set_trace()
X_batch = tf.train.batch([X_train], batch_size=3, capacity=3, enqueue_many=True)

init = tf.global_variables_initializer()

# execution
sess = tf.InteractiveSession()
sess.run(init)
tf.train.start_queue_runners(sess)
print sess.run(X_batch)
