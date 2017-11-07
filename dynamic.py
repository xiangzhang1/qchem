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

from scipy.optimize import minimize

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


# MlVaspSpeed
# ==============================================================================

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
            ('log', FunctionTransformer(func=np.log, inverse_func=np.exp)),      # reduce information to reasonable
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
            y_B = tf.layers.dense(y_B_2, units=1)
        with tf.variable_scope('C'):
            y_C = tf.layers.dense(X[:, 8:], units=1)
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
                batch_idx = np.random.choice(_X.shape[0]-5, size=batch_size)
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
            _y = sess.run(y)
        _y_inverse = self.y_pipeline.inverse_transform(_y)
        return _y_inverse






# MlPbSOpt
# ==============================================================================

class MlPbSOpt(object):

    def __init__(self):
        # data
        self._X = []
        self._y0 = []
        # pipeline
        self.X_pipeline = StandardScaler()
        self.y_pipeline = StandardScaler()
        # ann. what a pity.
        self.path = shared.SCRIPT_DIR + str.upper(self.__class__.__name__)
        tf.reset_default_graph()
        self.ann(tf.placeholder(tf.float32, shape=(None, 126)), training=False)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, self.path)

    def err_after_tf(self, m, ccoor, origin, a):   # error after transformation m = [dx, dy, dz, theta, phi, xi]
        dx, dy, dz, theta, phi, xi = m
        ## after rotation
        M = shared.euler2mat(theta, phi, xi).T    # order doesn't matter so I can do x . M
        ccoor_prime = np.dot(ccoor + [dx, dy, dz], M)
        ## error after rotation, in fcoor. note that ccoor is the main format.
        fcoor_prime = (ccoor_prime - origin) / a
        err = np.linalg.norm(fcoor_prime - np.around(fcoor_prime))
        return err

    def parse_obj(self, vasp):
        matrices = []
        for mirror_x in [-1, 1]:
            for mirror_y in [-1, 1]:
                for mirror_z in [-1, 1]:
                    for swap_matrix in ([[1,0,0],[0,0,1],[0,1,0]], [[0,1,0],[1,0,0],[0,0,1]], [[0,0,1],[0,1,0],[1,0,0]]):
                        transf = np.dot(np.diag([mirror_x, mirror_y, mirror_z]), swap_matrix)
                        self._parse_obj(np.dot(vasp.optimized_cell.ccoor, transf), vasp.optimized_cell.stoichiometry['Pb'] - vasp.optimized_cell.stoichiometry['S'])


    def _parse_obj(self, ccoor, off_stoi):
        a = 6.01417 / 2
        # coordination system
        origin = ccoor[0]
        m0 = np.random.uniform(-0.3, 0.3, 6)
        res = minimize(fun=self.err_after_tf, x0=m0, args=(ccoor, origin, a))  # find the absolute-neutral system
        m = res.x ; dx, dy, dz, theta, phi, xi = m
        err = res.fun
        M = shared.euler2mat(theta, phi, xi).T  # use that system
        ccoor = np.dot(ccoor + [dx, dy, dz], M)
        # each
        for i, c in enumerate(ccoor):
            # dx_self
            dx_i = (c - np.around(c / a) * a)[0]    # scalar
            # dx_jkl in order, i_jkl in order
            list_dx_jkl = []
            list_i_jkl = []
            for j in range(-2, 3):
                for k in range(-2, 3):
                    for l in range(-2, 3):
                        list_c_jkl = [c_jkl for c_jkl in ccoor if all(c + np.array([j-0.5, k-0.5, l-0.5]) * a < c_jkl) and all(c + np.array([j+0.5, k+0.5, l+0.5]) * a > c_jkl)]
                        list_i_jkl.append(1 if list_c_jkl else 0)
                        c_jkl = list_c_jkl[0] if list_c_jkl else [0, 0, 0]
                        dx_jkl = (c_jkl - np.around(c_jkl / a) * a)[0]  # scalar
                        list_dx_jkl.append(dx_jkl)
        # add to database, together with symmetrics
        self._X.append(list_i_jkl + [off_stoi])
        self._y0.append([dx_i])



    def ann(self, X, training):
        y1 = bel(X, units=10, training=training)
        y2 = bel(y1, units=3, training=training)
        y = tf.layers.dense(y2, units=1)
        return y


    def train(self):
        n_epochs = 1000
        batch_size = 72
        learning_rate = 0.01
        # pipeline
        _X = self.X_pipeline.fit_transform(self._X)
        _y0 = self.y_pipeline.fit_transform(self._y0)
        # batch
        # ann
        tf.reset_default_graph()
        _X_batch = tf.placeholder(tf.float32, shape=[None, 126])
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
                batch_idx = np.random.choice(_X.shape[0]-5, size=batch_size)
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
            _y = sess.run(y)
        _y_inverse = self.y_pipeline.inverse_transform(_y)
        return _y_inverse
