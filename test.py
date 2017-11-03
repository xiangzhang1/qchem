#!/usr/bin/python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import tensorflow as tf
import numpy as np
import time
import dill as pickle
import IPython

import shared


# Parameters
n_X_A, n_hidden_A1, n_hidden_A2, n_y_A = 8, 8, 4, 1
n_X_B, n_hidden_B1, n_hidden_B2, n_y_B = 2, 4, 4, 1
n_X, n_y = 10, 1
path = '/home/xzhang1/src/qchem/qchem/data/MLVASPMEMORY'

n_epochs = 500
batch_size = 32
learning_rate = 0.0001
training = True
reuse=False



# Data
data = []
with open("/home/xzhang1/m_cpu_config.log", "r") as if_:
    lines = if_.readlines()
    for line in lines:
        if line and not line.startswith('#') and len(line.split())==len(lines[1].split()):
            data.append( np.float_(line.split()) )
data = np.float_(data)
data /= np.float32([10**8, 10**8, 10**8, 10**8, 100, 1000, 1, 10, 10**3, 10**3, 1])
X_data = data[:, :-3]
X_data = np.concatenate((X_data, [[0,2]]*X_data.shape[0]), axis=1)
y_data = data[:, -2:-1]

data = np.concatenate((X_data, y_data), axis=1)
data = np.float32(data)





# ANN: construct

dataset = tf.contrib.data.Dataset.from_tensor_slices((data[:, :-1], data[:, -1:]))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
iterator = dataset.make_one_shot_iterator()
X_batch, y_batch = iterator.get_next()

X_A = tf.slice(X_batch, [0, 0], [-1, n_X_A], name='X_A')
X_B = tf.slice(X_batch, [0, n_X_A], [-1, -1], name='X_B')

hidden_B1 = tf.layers.dense(X_B, units=4, activation=tf.nn.relu)
hidden_B2 = tf.layers.dense(hidden_B1, units=4, activation=tf.nn.relu)
y_B = tf.layers.dense(hidden_B2, units=1)

hidden_A1 = tf.layers.dense(X_A, n_hidden_A1, activation=tf.nn.relu)
hidden_A2 = tf.layers.dense(hidden_A1, n_hidden_A2, activation=tf.nn.relu)
y_A = tf.layers.dense(hidden_A2, 1)

y = tf.multiply(y_A, y_B, name='y')


loss = tf.nn.l2_loss(y - y_batch)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(loss)


# ANN: execute
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        for _ in range(data.shape[0] // batch_size):
            # _X_batch, _y_batch, _X_A, _X_B, _y_A, _y_B, _y, _ = sess.run([X_batch, y_batch, X_A, X_B, y_A, y_B, y, training_op])
            _X_batch, _y_batch, _y, _loss, _ = sess.run([X_batch, y_batch, y, loss, training_op])
        if epoch % 1 == 0:
            # print '-' * 50
            # print 'Epoch %s \n\n X_batch %s \n\n y_batch %s \n\n X_A %s \n\n X_B %s \n\n y_A %s \n\n y_B %s \n\n y %s' %(epoch, _X_batch, _y_batch, _X_A, _X_B, _y_A, _y_B, _y)
            # print 'X_batch %s \n\n y_batch %s \n\n y %s \n\n' %(_X_batch, _y_batch, _y)
            print '%s %s' %(epoch, _loss)
