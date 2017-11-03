#!/usr/bin/python
import os
import tensorflow as tf
import numpy as np
import time
import dill as pickle
import IPython

import shared

# data
data = []
with open("/home/xzhang1/m_cpu_config.log", "r") as if_:
    lines = if_.readlines()
    for line in lines:
        if line and not line.startswith('#') and len(line.split())==len(lines[1].split()):
            data.append( np.float_(line.split()) )
data = np.float_(data)
data /= np.float32([10**8, 10**8, 10**8, 10**8, 1, 1000, 1, 10, 10**3, 10**3, 1])
X_data = data[:, :-3]
X_data = np.concatenate((X_data, [[0,2]]*X_data.shape[0]), axis=1)
y_data = data[:, -2:-1]

# ANN: construct
self.data = np.concatenate((X_data, y_data), axis=1)



n_epochs = 1024
batch_size = 256
learning_rate = 0.03
training = True
reuse=False

# data
data = np.float32(self.data)

# ANN: construct
tf.reset_default_graph()
X_batch, y_batch = self.iterator(data[:, :-1], data[:, -1:], batch_size=batch_size)
with tf.variable_scope('diverge_AB', reuse=reuse):
    X_A = tf.slice(X_batch, [0, 0], [-1, self.n_X_A], name='X_A')
    X_B = tf.slice(X_batch, [0, self.n_X_A], [-1, -1], name='X_B')

with tf.variable_scope('ann_B', reuse=reuse):
    hidden_B1 = tf.nn.elu(tf.layers.batch_normalization(tf.layers.dense(X_B, self.n_hidden_B1), training=training, momentum=0.9))
    hidden_B2 = tf.nn.elu(tf.layers.batch_normalization(tf.layers.dense(hidden_B1, self.n_hidden_B2), training=training, momentum=0.9))
    y_B = tf.multiply(tf.layers.dense(hidden_B2, 1, activation=tf.sigmoid), 6, name='y_B')

with tf.variable_scope('ann_A', reuse=reuse):
    hidden_A1 = tf.nn.elu(tf.layers.batch_normalization(tf.layers.dense(X_A, self.n_hidden_A1), training=training, momentum=0.9))
    hidden_A2 = tf.nn.elu(tf.layers.batch_normalization(tf.layers.dense(hidden_A1, self.n_hidden_A2), training=training, momentum=0.9))
    y_A = tf.layers.dense(hidden_A2, 1, name='y_A')

with tf.variable_scope('converge_AB', reuse=reuse):
    y = tf.multiply(y_A, y_B, name='y')
#
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
loss = tf.nn.l2_loss(y - y_batch)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(loss)  # remember to wipe your ass!
saver = tf.train.Saver()

# ANN: execute
with tf.Session() as sess:
    saver.restore(sess, self.path)
    for epoch in range(n_epochs):
        for _ in range(data.shape[0] // batch_size):
            sess.run([update_ops, training_op])
        if epoch % 50 == 0:
            print 'Epoch %s, loss %s' %(epoch, sess.run(loss))
    saver.save(sess, self.path)
