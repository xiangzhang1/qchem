#!/usr/bin/python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import numpy as np

import tensorflow as tf

from functools import partial
batch_norm_layer = partial(tf.layers.batch_normalization,
                      training=training, momentum=0.9)


'''VASP memory usage.
X: projector_real, projector_reciprocal, wavefunction, array_grid, natoms, volume, npar, ncore_total; lsorbit=0|1, isym=-1|0|1|2.
y: memory in byte'''


# Data
# ==========================
X_data = None
y_data = None

# Batch
# ==========================
batch_size = 30
with tf.name_scope('batch'):
    X_batch, y_batch = tf.train.shuffle_batch([X_data, y_data], batch_size=batch_size, capacity=tf.shape(X_raw)[0], min_after_dequeue=tf.shape(X_raw)[0]-batch_size)

# NN
# ==========================
# Construction
n_inputs_A = 8
n_hidden_A1 = 8
n_hidden_A2 = 4
n_outputs_A = 1

n_inputs_B = 2
n_hidden_B1 = 4
n_hidden_B2 = 4
n_outputs_B = 1

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_inputs_A + n_inputs_B), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
training = tf.placeholder_with_default(False, shape=(None), name='training')

with tf.name_scope('diverge_AB'):
    X_A = X[:, :n_inputs_A]
    X_B = X[:, n_inputs_A:]

with tf.name_scope('DNN_B'):
    hidden_B1 = tf.layers.dense(X_B, n_hidden_B1, name='hidden_B1')
    bn_B1 = batch_norm_layer(hidden_B1)
    bn_B1_act = tf.nn.elu(bn_B1)
    hidden_B2 = tf.layers.dense(bn_B1_act, n_hidden_B2, name='hidden_B2')
    bn_B2 = batch_norm_layer(hidden_B2)
    bn_B2_act = tf.nn.elu(bn_B2)
    output_B = tf.layers.dense(bn_B2_act, 1, name='output_B', activation=tf.sigmoid) * 6

with tf.name_scope('loss_train_B'):
    y_B = tf.placeholder(tf.float32, shape=(None, 1), name='y_B')
    loss_B = tf.nn.l2_loss(output_B - y_B)
    optimizer_B = tf.train.GradientDescentOptimizer(learning_rate)
    update_ops_B = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops_B):
        training_op_B = optimizer_B.minimize(loss_B)

with tf.name_scope('DNN_A'):
    hidden_A1 = tf.layers.dense(X_A, n_hidden_A1, name='hidden_A1')
    bn_A1 = batch_norm_layer(hidden_A1)
    bn_A1_act = tf.nn.elu(bn_A1)
    hidden_A2 = tf.layers.dense(bn_A1_act, n_hidden_A2, name='hidden_A2')
    bn_A2 = batch_norm_layer(hidden_A2)
    bn_A2_act = tf.nn.elu(bn_A2)
    output_A = tf.layers.dense(bn_A2_act, 1, name='output_A')

with tf.name_scope('converge_AB'):
    output = tf.multiply(output_A, output_B, name='output')

with tf.name_scope('loss_train'):
    loss = tf.nn.l2_loss(output_B - y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()


# execution
sess = tf.InteractiveSession()
init.run()

# separately train B
n_epochs = 100
X_train = np.float_([
    [0, 2],
    [1, 2],
    [0, 0],
    [1, 0],
    [0, -1],
    [1, -1]
])
y_train = np.float_([
    [1],
    [2],
    [2],
    [3],
    [3.3],
    [4.5]
])

for epoch in range(n_epochs):
    sess.run(training_op_B, feed_dict={X_B: X_train, y_B: y_train, training: True})
    print 'Epoch %s, mse %s' %(epoch, loss_B.eval(feed_dict={X_B: X_train, y_B: y_train, training: False}))

output_B.eval(feed_dict={X_B: X_train, y_B: y_train, training: True})

# train whole
n_epochs = 40
batch_size = 50

data = []
with open("/home/xzhang1/m_cpu_config.log", "r") as if_:
    lines = if_.readlines()
    for line in lines:
        if line and not line.startswith('#') and len(line.split())==len(lines[1].split()):
            data.append( np.float_(line.split()) )
data = np.float_(data)
data[:, -2:-1] *= 10**6    # SI unit
X_train = data[:1000, :-3]
X_train = np.concatenate((X_train, [[0,2]]*X_train.shape[0]), axis=1)
y_train = data[:1000, -2:-1]
X_test = data[1000:, :-3]
X_train = np.concatenate((X_test, [[0,2]]*X_test.shape[0]), axis=1)
y_test = data[1000:, -2:-1]

for epoch in range(n_epochs):
    for iteration in range(data.shape[0] // batch_size):
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
    print 'Epoch %s, loss %s' %(loss.run(feed_dict={X: data_test[:, :-3], y: data_test[:, -2], training: False}))
