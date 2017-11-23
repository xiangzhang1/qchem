'''
First example, from Hands-on book
'''

import tensorflow as tf
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # GPU is slower than CPU on research-scale problems
os.environ["CUDA_VISIBLE_DEVICES"] = ''

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

from datetime import datetime

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# scale
X_scaler = StandardScaler()
scaled_housing_data_plus_bias = housing_data_plus_bias
scaled_housing_data_plus_bias[:, 1:] = X_scaler.fit_transform(housing_data_plus_bias[:, 1:])

# construction
n_epochs = 1000
learning_rate = 0.01
batch_size = 1000
n_batches = int(np.ceil(m / batch_size))

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

with tf.name_scope("predict") as scope:
    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
with tf.name_scope("optimize") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)
with tf.name_scope("log") as scope:
    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

init = tf.global_variables_initializer()

# execution

def fetch_batch(epoch, batch_index, batch_size):
    # load the data from disk
    X = scaled_housing_data_plus_bias
    y = housing.target.reshape(-1, 1)
    min_idx = batch_index * batch_size
    max_idx = (batch_index + 1) * batch_size
    X_batch = X[min_idx: max_idx]
    y_batch = y[min_idx: max_idx]
    return X, y

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
                print step, mse.eval(feed_dict={X: X_batch, y: y_batch})
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
