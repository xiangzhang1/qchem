#!/usr/bin/python
# THERE IS NO FREE POWER.
import os
import tensorflow as tf
import numpy as np
import time
import dill as pickle
import IPython

import shared

# save, load
# ==============================================================================

def save(obj, middlename):      # Note! Where defined, above which module pickled.
    filepath = shared.SCRIPT_DIR + '/data/dynamic.%s.pickle.'%(middlename) + time.strftime('%Y%m%d%H%M%S')
    with open(filepath,'wb') as dumpfile:
        pickle.dump(obj, dumpfile)

def load(middlename, datetime=None):
    filepath = shared.SCRIPT_DIR + '/data/' + sorted([x for x in os.listdir(shared.SCRIPT_DIR + '/data/') if x.startswith('dynamic.%s.pickle.%s'%(middlename, datetime if datetime else ''))])[-1]
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def global_load():
    global NODES
    NODES = load('NODES')

# NODES
# ==============================================================================

NODES = {}

# ML
# ==============================================================================

MLS = {}



# MlVaspMemory
# ==============================================================================

class MlVaspMemory(object):

    def iterator(self, X_data, y_data, n_epochs, batch_size):
        with tf.variable_scope('iterator'):
            dataset = tf.contrib.data.Dataset.from_tensor_slices((X_data, y_data))
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(batch_size)
            dataset = dataset.repeat(n_epochs)
            iterator = dataset.make_one_shot_iterator()
            X_batch, y_batch = iterator.get_next()
        return X_batch, y_batch


    def ann(self, X, training, reuse):
        with tf.variable_scope('diverge_AB', reuse=reuse):
            X_A = tf.slice(X, [0, 0], [-1, self.n_X_A], name='X_A')
            X_B = tf.slice(X, [0, self.n_X_A], [-1, -1], name='X_B')

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

        return y


    def __init__(self):
        # Paramters
        self.n_X_A, self.n_hidden_A1, self.n_hidden_A2, self.n_y_A = 8, 8, 4, 1
        self.n_X_B, self.n_hidden_B1, self.n_hidden_B2, self.n_y_B = 2, 4, 4, 1
        self.n_X, self.n_y = 10, 1
        self.path = shared.SCRIPT_DIR + str.upper(self.__class__.__name__)

        # initialize X_data
        self.data = []

        # initialize ANN
        tf.reset_default_graph()
        self.ann(tf.placeholder(tf.float32, shape=(None, self.n_X)),
                 training=True, reuse=False)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, self.path)


    def init_data(self):

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



    def fit_B(self):
        n_epochs = 100
        learning_rate = 0.01

        # data
        data = np.float_([[0, 2, 1], [1, 2, 2], [0, 0, 2], [1, 0, 3], [0, -1, 3.3], [1, -1, 4.5]])

        # ann_B: construct
        tf.reset_default_graph()
        self.ann(tf.placeholder(tf.float32, shape=(None, self.n_X)),
                 training=True, reuse=False)
        X = tf.get_default_graph().get_tensor_by_name("diverge_AB/X_B:0")
        y = tf.get_default_graph().get_tensor_by_name("ann_B/y_B:0")
        y_ = tf.placeholder(tf.float32, shape=(None, self.n_y_B), name='y_')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='ann_B')
        loss = tf.nn.l2_loss(y - y_)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        with tf.control_dependencies(update_ops):
            training_op = optimizer.minimize(loss)
        saver = tf.train.Saver()

        # ann_B: run
        with tf.Session() as sess:
            saver.restore(sess, self.path)
            for epoch in range(n_epochs):
                sess.run(training_op, feed_dict={X: data[:, :-1], y_: data[:, -1:]})
            print 'fit_B complete. Loss: %s' %(loss.eval(feed_dict={X: data[:, :-1], y_: data[:, -1:]}))
            saver.save(sess, self.path)



    def fit(self):
        n_epochs = 512
        batch_size = 96
        learning_rate = 0.01

        # data
        data = np.float32(self.data)

        # ANN: construct
        tf.reset_default_graph()
        X_batch, y_batch = self.iterator(data[:, :-1], data[:, -1:], n_epochs=n_epochs, batch_size=batch_size)
        y = self.ann(X_batch, training=True, reuse=False)
        #
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        loss = tf.nn.l2_loss(y - y_batch)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        with tf.control_dependencies(update_ops):     # Wipe your ass immediately!
            training_op = optimizer.minimize(loss)
        saver = tf.train.Saver()

        # ANN: execute
        with tf.train.MonitoredTrainingSession() as sess:
            saver.restore(sess, self.path)
            while not sess.should_stop():
                sess.run(training_op)
                print '3',
            saver.save(sess, self.path)

        with tf.Session() as sess:
            saver.restore(sess, self.path)
            print 'Loss for newest data point: %s' %(loss.eval(feed_dict={X_batch: data[-1:, :-1], y_batch: data[-1:, -1:]}))


    def predict(self, X_new):
        # data
        X_new = [X_new]

        # ANN: construct
        tf.reset_default_graph()
        y = self.ann(X_new, training=False, reuse=False)
        saver = tf.train.Saver()

        # ANN: run
        with tf.Session() as sess:
            saver.restore(sess, self.path)
            return float(y)
