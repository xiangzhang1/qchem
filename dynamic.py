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
    global NODES, MLS
    NODES = load('NODES')
    MLS = load('MLS')

def global_save():
    global NODES, MLS
    save(NODES, 'NODES')
    save(MLS, 'MLS')


# NODES
# ==============================================================================

NODES = {}

# ML
# ==============================================================================

MLS = {}



# MlVaspMemory
# ==============================================================================

class MlVaspMemory(object):

    def __init__(self):

        # X_data
        self._X, self._y0 = [], []

        # scaler_constant
        self.X_scaler_constant = np.float32([10**8, 10**8, 10**8, 10**8, 100, 1000, 1, 10, 1, 1])
        self.y_scaler_constant = 10.0**9

        # Paramters
        self.n_X_A, self.n_hidden_A1, self.n_hidden_A2, self.n_y_A = 8, 8, 4, 1
        self.n_X_B, self.n_hidden_B1, self.n_hidden_B2, self.n_y_B = 2, 4, 4, 1
        self.n_X, self.n_y = 10, 1
        self.path = shared.SCRIPT_DIR + 'data/' + str.upper(self.__class__.__name__)

        # first ANN dump
        tf.reset_default_graph()
        self.ann(tf.placeholder(tf.float32, shape=(None, self.n_X)),
                 training=True, reuse=False)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, self.path)


    def iterator(self, _X, _y0, batch_size):
        with tf.variable_scope('iterator'):
            dataset = tf.contrib.data.Dataset.from_tensor_slices((_X, _y0))
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(batch_size)
            dataset = dataset.repeat()
            iterator = dataset.make_one_shot_iterator()
            X_batch, y0_batch = iterator.get_next()
        return X_batch, y0_batch


    def ann(self, X, training, reuse):
        with tf.variable_scope('diverge_AB', reuse=reuse):
            X_A = tf.slice(X, [0, 0], [-1, self.n_X_A], name='X_A')
            X_B = tf.slice(X, [0, self.n_X_A], [-1, -1], name='X_B')

        with tf.variable_scope('ann_B', reuse=reuse):
            hidden_B1 = tf.layers.dense(X_B, self.n_hidden_B1)
            hidden_B1_normalized = tf.layers.batch_normalization(hidden_B1, training=training, momentum=0.9)
            hidden_B1_act = tf.nn.elu(hidden_B1_normalized)
            hidden_B1_dropout = tf.layers.dropout(hidden_B1_act, rate=0.1, training=training)
            hidden_B2 = tf.layers.dense(hidden_B1_dropout, self.n_hidden_B2)
            hidden_B2_normalized = tf.layers.batch_normalization(hidden_B2, training=training, momentum=0.9)
            hidden_B2_act = tf.nn.elu(hidden_B2_normalized)
            hidden_B2_dropout = tf.layers.dropout(hidden_B2_act, rate=0.1, training=training)
            hidden_B3 = tf.layers.dense(hidden_B2_dropout, 1, activation=tf.sigmoid)
            y_B = tf.multiply(hidden_B3, 6, name='y_B')

        with tf.variable_scope('ann_A', reuse=reuse):
            hidden_A1 = tf.layers.dense(X_A, self.n_hidden_A1)
            hidden_A1_normalized = tf.layers.batch_normalization(hidden_A1, training=training, momentum=0.9)
            hidden_A1_act = tf.nn.elu(hidden_A1_normalized)
            hidden_A1_dropout = tf.layers.dropout(hidden_A1_act, rate=0.1, training=training)
            hidden_A2 = tf.layers.dense(hidden_A1_dropout, self.n_hidden_A2)
            hidden_A2_normalized = tf.layers.batch_normalization(hidden_A2, training=training, momentum=0.9)
            hidden_A2_act = tf.nn.elu(hidden_A2_normalized)
            hidden_A2_dropout = tf.layers.dropout(hidden_A2_act, rate=0.1, training=training)
            hidden_A3 = tf.layers.dense(hidden_A2_dropout, 1, activation=tf.sigmoid)
            y_A = tf.multiply(hidden_A3, 6, name='y_A')

        with tf.variable_scope('converge_AB', reuse=reuse):
            y = tf.multiply(y_A, y_B, name='y')

        return y



    def init_data(self):

        # data
        data = []
        with open("/home/xzhang1/m_cpu_config.log", "r") as if_:
            lines = if_.readlines()
            for line in lines:
                if line and not line.startswith('#') and len(line.split())==len(lines[1].split()):
                    data.append( np.float_(line.split()) )
        data = np.float_(data)
        data[:, -3:-1] *= 10**6     # not converting! It's raw SI.
        self._X = np.concatenate((data[:, :-3], [[0,2]] * data.shape[0]), axis=1)
        self._y0 = data[:, -2:-1]



    def train_B(self):
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
        y0 = tf.placeholder(tf.float32, shape=(None, self.n_y_B), name='y0')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='ann_B')
        loss = tf.reduce_mean(tf.squared_difference(y, y0))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)    # remember to wipe your ass!
        saver = tf.train.Saver()

        # ann_B: run
        with tf.Session() as sess:
            saver.restore(sess, self.path)
            for epoch in range(n_epochs):
                sess.run([update_ops, training_op], feed_dict={X: data[:, :-1], y0: data[:, -1:]})
            print self.__class__.__name__ + ':train_B complete. Evaluation not implemented. '
            saver.save(sess, self.path)



    def train(self):
        n_epochs = 1000
        batch_size = 32
        learning_rate = 0.001

        # scaler
        _X_scaled = np.float32(self._X) / self.X_scaler_constant
        _y0_scaled = np.float32(self._y0) / self.y_scaler_constant

        # ANN: construct
        tf.reset_default_graph()
        X_batch, y0_batch = self.iterator(_X_scaled, _y0_scaled, batch_size=batch_size)
        y = self.ann(X_batch, training=True, reuse=False)
        #
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        loss = tf.reduce_mean(tf.squared_difference(y, y0_batch))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)  # remember to wipe your ass!
        saver = tf.train.Saver()

        # ANN: execute
        print self.__class__.__name__ + ': training with dropout:'
        with tf.Session() as sess:
            saver.restore(sess, self.path)
            for epoch in range(n_epochs):
                for _ in range(_X_scaled.shape[0] // batch_size):
                    _loss, _, _ = sess.run([loss, update_ops, training_op])
                if epoch % 50 == 0:
                    print self.__class__.__name__ + ': epoch %s, loss %s' %(epoch, _loss)
            saver.save(sess, self.path)
            print self.__class__.__name__ + ': training complete.'

        # Test
        # scaler
        _X_scaled = np.float32(self._X[-3:]) / self.X_scaler_constant
        _y0_scaled = np.float32(self._y0[-3:]) / self.y_scaler_constant
        # ANN
        tf.reset_default_graph()
        y = self.ann(_X_scaled, training=False, reuse=False)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.path)
            _y = y.eval()

        print self.__class__.__name__ + ': Evaluation of ANN. \n X is %s.\n y0 is %s.\n y is %s.\n' %(_X_scaled, _y0_scaled, _y)


    def predict(self, _X):
        # scale
        _X_scaled = np.float32(np.atleast_2d(_X)) / self.X_scaler_constant

        # ANN
        tf.reset_default_graph()
        y = self.ann(_X_scaled, training=False, reuse=False)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.path)
            _y = y.eval()

        return _y * self.y_scaler_constant
