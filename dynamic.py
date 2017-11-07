#!/usr/bin/python
import os
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer, StandardScaler
from shared import LabelBinarizerPipelineFriendly
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np
import time
import dill as pickle
import IPython
from tqdm import tqdm

import shared

# save, load
# ==============================================================================

def save(obj, middlename):      # Note! Where defined, above which module pickled.
    filepath = shared.SCRIPT_DIR + '/data/dynamic.%s.pickle.'%(middlename) + time.strftime('%Y%m%d%H%M%S')
    with open(filepath,'wb') as dumpfile:
        pickle.dump(obj, dumpfile, protocol=pickle.HIGHEST_PROTOCOL)

def load(middlename, datetime=None):
    filepath = shared.SCRIPT_DIR + '/data/' + sorted([x for x in os.listdir(shared.SCRIPT_DIR + '/data/') if x.startswith('dynamic.%s.pickle.%s'%(middlename, datetime if datetime else ''))])[-1]
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def global_load():
    global NODES, MLS
    NODES = load('NODES')
    MLS = load('MLS')
    print 'global_load called'

def global_save():
    global NODES, MLS
    save(NODES, 'NODES')
    save(MLS, 'MLS')
    print 'global_save called'


# NODES
# ==============================================================================

NODES = {}

# ML
# ==============================================================================

MLS = {}

def bel(X, units, training):
    '''Returns a Batch-normalized, Elu-activated Layer.
    If regression=True,
    Reuse is not considered.'''
    h1 = tf.layers.dense(X, units=units)
    h1_normalized = tf.layers.batch_normalization(h1, training=training, momentum=0.5)
    h1_act = tf.nn.elu(h1_normalized)
    h1_dropout = tf.layers.dropout(h1_act, rate=0.3)
    return h1_act


class MlVaspSpeed(object):

    def __init__(self):
        # data
        self._X = []
        self._y0 = []
        # pipeline
        self.X_pipeline = Pipeline([
            ('cast_to_array', FunctionTransformer(func=np.array)),
            ('split', FeatureUnion(transformer_list=[
                ('A', Pipeline([
                    ('slicer', FunctionTransformer(func=lambda X: X[:, :5])),
                    ('caster', FunctionTransformer(func=np.float32)),
                    ('scaler', StandardScaler())
                ])),
                ('B', Pipeline([
                    ('slicer', FunctionTransformer(func=lambda X: X[:, 5:8])),
                    ('caster', FunctionTransformer(func=np.float32)),
                    ('scaler', StandardScaler())
                ])),
                ('C', Pipeline([
                    ('slice_flatten', FunctionTransformer(func=lambda X: X[:, 8].flatten())),
                    ('labeler', LabelBinarizerPipelineFriendly()),
                    ('padder', FunctionTransformer(func=lambda X: np.hstack((X, np.zeros((X.shape[0], 4-X.shape[1]))))))
                ]))
            ])),
            ('cast_to_float32', FunctionTransformer(func=np.float32))
        ])
        self.y_pipeline = Pipeline([
            # ('log', FunctionTransformer(func=np.log)),      # reduce information to reasonable
            ('scaler', StandardScaler())
        ])
        # ann. what a pity.
        self.path = shared.SCRIPT_DIR + str.upper(self.__class__.__name__)
        tf.reset_default_graph()
        self.ann(tf.placeholder(tf.float32, shape=(None, 12)), training=False)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, self.path)

    def parse_obj(self, vasp, makeparam):
        # OUTPUT
        # ------
        # preliminary checks
        if vasp.moonphase() != 2:
            raise shared.CustomError(self.__class__.__name__ + '.warning: vasp moonphase is not 2. skipped collect data.')
        if not os.path.isfile(vasp.path+'/OUTCAR'):
            raise shared.CustomError(self.__class__.__name__ + '.warning: no OUTCAR found. skipped collect data.')
        # parse outcar for time (s) / #elecstep
        os.chdir(vasp.path)
        with open(vasp.path + '/OUTCAR', 'r') as f:
            lines = f.readlines()
            # total time
            line = [l for l in lines if 'Total CPU time used' in l]
            if not line:
                raise shared.CustomError(self.__class__.__name__ + '.warning: no Total CPU time line found. skipped collect data.')
            total_time = float(line[-1].split()[-1])
            if total_time < 1:
                raise shared.CustomError(self.__class__.__name__ + '.warning: total time does not feel right. skipped colelct data.')
            # number of ionic steps
            iteration_lines = [l for l in lines if 'Iteration' in l]
            number_elec_steps = len(iteration_lines)
            # time per electronic step
            time_elec_step = total_time / number_elec_steps
        # INPUT
        # -----
        gen = vasp.gen
        cell = vasp.cell
        self._X.append([
            makeparam.projector_real + makeparam.projector_reciprocal,
            makeparam.wavefunction,
            makeparam.arraygrid,
            vasp.gen.cell.natoms(),
            np.dot(np.cross(gen.cell.base[0], gen.cell.base[1]), gen.cell.base[2]),
            # platform-specific
            int(gen.getkw('ncore_node')),
            int(gen.getkw('nnode')),
            int(gen.getkw('ncore')),
            gen.getkw('platform')
        ])
        self._y0.append([time_elec_step])   # put it here so that no inconsistency will happen

    def ann(self, X, training):
        with tf.variable_scope('A'):
            y_A_1 = bel(X[:, :5], units=3, training=training)
            y_A_2 = bel(y_A_1, units=3, training=training)
            y_A = tf.layers.dense(y_A_2, units=1)
        with tf.variable_scope('B'):
            y_B_1 = bel(X[:, 5:8], units=3, training=training)
            y_B_2 = bel(y_B_1, units=3, training=training)
            y_B = tf.layers.dense(y_B_2, units=1, activation=tf.sigmoid)
        with tf.variable_scope('C'):
            y_C = tf.layers.dense(X[:, 8:], units=1, activation=tf.sigmoid)
        with tf.variable_scope('converge'):
            y_1 = tf.concat([y_A, y_B, y_C], axis=1)
            y_2 = bel(y_1, units=3, training=training)
            y_3 = bel(y_2, units=3, training=training)
            y = tf.layers.dense(y_3, units=1)
        return y

    def train(self):
        n_epochs = 5000
        batch_size = 69
        learning_rate = 0.01
        # pipeline
        _X = self.X_pipeline.fit_transform(self._X)
        _y0 = self.y_pipeline.fit_transform(self._y0)
        # batch
        # ann
        tf.reset_default_graph()
        _X_batch = tf.placeholder(tf.float32, shape=[None, 12])
        _y0_batch = tf.placeholder(tf.float32, shape=[None, 1])
        y = self.ann(_X_batch, training=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        loss = tf.reduce_mean(tf.squared_difference(y, _y0_batch))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
        saver = tf.train.Saver()
        # train
        print self.__class__.__name__ + '.train: training started.'
        with tf.Session() as sess:
            saver.restore(sess, self.path)
            for i in range(n_epochs * _X.shape[0] / batch_size):
                batch_idx = np.random.choice(_X.shape[0], size=batch_size)
                _loss, _, _ = sess.run([loss, update_ops, training_op], feed_dict={_X_batch: _X[batch_idx], _y0_batch: _y0[batch_idx]})
                if i % 100 == 0:
                    print 'step %s, loss %s' %(i, _loss)
            saver.save(sess, self.path)

        # evaluate
        _X = self._X[-10:]
        _y0 = self._y0[-10:]
        _y = self.predict(_X)
        print self.__class__.__name__ + '.train: training finished. evaluation on last item: actual %s, predicted %s' %(_y0, _y)


    def predict(self, X):
        # pipeline
        X = self.X_pipeline.fit_transform(X)
        # ann
        tf.reset_default_graph()
        y = self.ann(tf.constant(X), training=False)
        saver = tf.train.Saver()
        # predict
        with tf.Session() as sess:
            saver.restore(sess, self.path)
            return sess.run(y)
