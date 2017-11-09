#!/usr/bin/python
import os
import numpy as np
import time
import dill as pickle
import IPython
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import minimize

# scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer, StandardScaler
from shared import LabelBinarizerPipelineFriendly
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import Pipeline, FeatureUnion

# pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

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
    '''Returns a Batch-normalized, Elu-activated Tensorflow layer.
    If regression=True,
    Reuse is not considered.'''
    h1 = tf.layers.dense(X, units=units)
    h1_normalized = tf.layers.batch_normalization(h1, training=training, momentum=0.5)
    h1_act = tf.nn.elu(h1_normalized)
    h1_dropout = tf.layers.dropout(h1_act, rate=0.1, training=training)
    return h1_act


# MlVaspSpeed
# ==============================================================================

class MlVaspSpeed(object):

    class Net(nn.Module):

        def __init__(self, bn_momentum, dropout_p):
            super(MlVaspSpeed.Net, self).__init__()
            self.lA1 = nn.Linear(5, 3)
            self.bnA1 = nn.BatchNorm1d(3, momentum=bn_momentum)
            self.dropoutA1 = nn.Dropout(p=dropout_p)
            self.lA2 = nn.Linear(3, 3)
            self.bnA2 = nn.BatchNorm1d(3, momentum=bn_momentum)
            self.dropoutA2 = nn.Dropout(p=dropout_p)
            self.lA3 = nn.Linear(3, 1)

            self.lB1 = nn.Linear(3, 3)
            self.bnB1 = nn.BatchNorm1d(3, momentum=bn_momentum)
            self.dropoutB1 = nn.Dropout(p=dropout_p)
            self.lB2 = nn.Linear(3, 3)
            self.bnB2 = nn.BatchNorm1d(3, momentum=bn_momentum)
            self.dropoutB2 = nn.Dropout(p=dropout_p)
            self.lB3 = nn.Linear(3, 1)

            self.lC1 = nn.Linear(4, 2)
            self.bnC1 = nn.BatchNorm1d(2, momentum=bn_momentum)
            self.dropoutC1 = nn.Dropout(p=dropout_p)
            self.lC2 = nn.Linear(2, 1)
            # self.lC3 = nn.Linear(4, 1)

            self.bn0 = nn.BatchNorm1d(3, momentum=bn_momentum)
            self.dropout0 = nn.Dropout(p=dropout_p)
            self.l1 = nn.Linear(3, 2)
            self.bn1 = nn.BatchNorm1d(2, momentum=bn_momentum)
            self.dropout1 = nn.Dropout(p=dropout_p)
            self.l2 = nn.Linear(2, 1)

        def forward(self, X):

            A = self.bnA1(self.dropoutA1(F.elu(self.lA1(X[:, :5]))))
            A = self.bnA2(self.dropoutA2(F.elu(self.lA2(A))))
            A = self.lA3(A)

            B = self.bnB1(self.dropoutB1(F.elu(self.lB1(X[:, 5:8]))))
            B = self.bnB2(self.dropoutB2(F.elu(self.lB2(B))))
            B = self.lB3(B)

            C = self.bnC1(self.dropoutC1(F.elu(self.lC1(X[:, 8:12]))))
            C = self.lC2(C)
            # C = self.lC3(X[:, 8:12])

            y = torch.cat((A, B, C), dim=1)
            y = self.bn0(self.dropout0(F.elu(y)))
            y = self.bn1(self.dropout1(F.elu(self.l1(y))))
            y = self.l2(y)
            # y = A * B * C

            return y


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
        self.net = MlVaspSpeed.Net(bn_momentum=0.7, dropout_p=0.2).cuda()


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

    def train(self, n_epochs=5000, batch_size=46, learning_rate=10E-5, optimizer_name='SGD'):
        # pipeline
        _X = self.X_pipeline.fit_transform(self._X)
        _y0 = self.y_pipeline.fit_transform(self._y0)
        # batch: random.choice
        # ann
        criterion = nn.MSELoss()
        optimizer = getattr(optim, optimizer_name)(self.net.parameters(), lr=learning_rate)
        # train
        self.net.train()
        for epoch in range(n_epochs):
            batch_idx = np.random.choice(range(_X.shape[0]), size=batch_size)
            X_batch= Variable(torch.FloatTensor(_X[batch_idx]), requires_grad=True).cuda()
            y0_batch = Variable(torch.FloatTensor(_y0[batch_idx]), requires_grad=False).cuda()
            y = self.net(X_batch)
            loss = criterion(y, y0_batch)
            optimizer.zero_grad()   # suggested trick
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print 'epoch %s, loss %s'%(epoch, loss.data.cpu().numpy()[0])

        # evaluate
        _X = self._X[-10:]
        _y0 = np.float32(self._y0[-10:]).flatten()
        _y = np.float32(self.predict(_X)).flatten()
        print self.__class__.__name__ + '.train: training finished. evaluation on last item: \n actual | predicted'
        for a, b in zip(_y0, _y):
            print a, b
        # create a metric
        a = np.zeros((_y0.shape[0], 2))
        a[:, 0] = _y0 / _y
        a[:, 1] = _y / _y0
        b = np.amax(a, axis=0)
        return np.sum(b)


    def predict(self, _X):
        # pipeline
        _X = self.X_pipeline.fit_transform(_X)
        # ann
        self.net.eval()
        y = self.net(Variable(torch.FloatTensor(_X), requires_grad=True))
        # pipeline
        _y_inverse = self.y_pipeline.inverse_transform(y.data.cpu().numpy())
        return _y_inverse


# inital training script for MlVaspSpeed




# MlPbSOpt
# ==============================================================================

# class MlPbSOpt(object):
#
#     def __init__(self):
#         # data
#         self._X = []
#         self._y0 = []
#         # pipeline
#         self.X_pipeline = StandardScaler()
#         self.y_pipeline = StandardScaler()
#         # ann. what a pity.
#         self.path = shared.SCRIPT_DIR + str.upper(self.__class__.__name__)
#         tf.reset_default_graph()
#         self.ann(tf.placeholder(tf.float32, shape=(None, 126)), training=True)
#         saver = tf.train.Saver()
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             saver.save(sess, self.path)
#
#     def parse_obj(self, vasp):
#         a = 6.01417/2
#         # matrices = []
#         # for mirror_x in [-1, 1]:
#         #     for mirror_y in [-1, 1]:
#         #         for mirror_z in [-1, 1]:
#         #             for swap_matrix in ([[1,0,0],[0,0,1],[0,1,0]], [[0,1,0],[1,0,0],[0,0,1]], [[0,0,1],[0,1,0],[1,0,0]]):
#         #                 transf = np.dot(np.diag([mirror_x, mirror_y, mirror_z]), swap_matrix)
#         #                 self._parse_obj(np.dot(vasp.optimized_cell.ccoor, transf), vasp.optimized_cell.stoichiometry['Pb'] - vasp.optimized_cell.stoichiometry['S'])
#
#         # first, we get the fcoor. no rotation need be taken into account.
#         ccoor = vasp.optimized_cell.ccoor
#         def e(r0, ccoor=ccoor, a=a):
#             fcoor = np.subtract(ccoor, r0) / a
#             return np.linalg.norm(fcoor - np.around(fcoor))
#         r0 = minimize(fun=e, x0=[0,0,0], bounds=[(-0.5*a,0.5*a) for _ in range(3)]).x
#         fcoor = np.subtract(ccoor, r0) / a
#
#         # second, we parse it.
#
#
#
#
#     def _parse_obj(self, ccoor, off_stoi):
#         a = 6.01417 / 2
#         # coordination system
#         origin = ccoor[20]
#         m0 = np.random.uniform(-0.01, 0.01, 6)
#         print 'optimizing...'
#         res = minimize(fun=self.err_after_tf, x0=m0, args=(ccoor, origin, a), method='Powell', tol=10E-5)  # find the absolute-neutral system
#         print 'optimized, result f(%s) = %s' %(res.x, res.fun)
#         dx, dy, dz, theta, phi, xi = res.x
#         err = res.fun
#         M = shared.euler2mat(theta, phi, xi).T  # use that system
#         ccoor = np.dot(ccoor - origin + [dx, dy, dz], M)
#         # each
#         for i, c in enumerate(tqdm(ccoor, desc='parsing ccoor')):
#             # dx_self
#             dx_i = (c - np.around(c / a) * a)[0]
#             list_i_jkl = []
#             for j, k, l in itertools.product(range(-2, 3), range(-2, 3), range(-2, 3)):
#                 list_c_jkl = [c_jkl for c_jkl in ccoor if all(c + np.array([j-0.5, k-0.5, l-0.5]) * a < c_jkl) and all(c + np.array([j+0.5, k+0.5, l+0.5]) * a > c_jkl)]
#                 list_i_jkl.append(1 if list_c_jkl else 0)
#             # add to database, together with symmetrics
#             self._X.append(list_i_jkl + [off_stoi])
#             self._y0.append([dx_i])
#
#
#     def ann(self, X, training):
#         y1 = bel(X, units=20, training=training)
#         y4 = bel(y1, units=4, training=training)
#         y = tf.layers.dense(y4, units=1)
#         return y
#
#
#     def train(self, n_epochs=500, batch_size=72, learning_rate=0.001):
#         # pipeline
#         _X = self.X_pipeline.fit_transform(self._X)
#         _y0 = self.y_pipeline.fit_transform(self._y0)
#         # batch
#         # ann
#         tf.reset_default_graph()
#         _X_batch = tf.placeholder(tf.float32, shape=[None, 126])
#         _y0_batch = tf.placeholder(tf.float32, shape=[None, 1])
#         y = self.ann(_X_batch, training=True)
#         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#         loss = tf.reduce_mean(tf.squared_difference(y, _y0_batch))
#         optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#         training_op = optimizer.minimize(loss)
#         saver = tf.train.Saver()
#         # train
#         print self.__class__.__name__ + '.train: training started.'
#         with tf.Session() as sess:
#             saver.restore(sess, self.path)
#             for i in tqdm(range(n_epochs * _X.shape[0] / batch_size)):
#                 batch_idx = np.random.choice(_X.shape[0]-100, size=batch_size)
#                 _loss, _, _ = sess.run([loss, training_op, update_ops], feed_dict={_X_batch: _X[batch_idx], _y0_batch: _y0[batch_idx]})
#                 # if i % 100 == 0:
#                 #     print 'step %s, loss %s' %(i, _loss)
#             saver.save(sess, self.path)
#
#         # evaluate
#         _X = self._X[-100:]
#         _y0 = self._y0[-100:]
#         _y = self.predict(_X)
#         print self.__class__.__name__ + '.train: training finished. evaluation on last item: actual %s, predicted %s' %(_y0, _y)
#         plt.scatter(_y0, _y)
#         plt.show()
#
#
#     def predict(self, _X):
#         # pipeline
#         _X_batch = self.X_pipeline.transform(_X)
#         # ann
#         tf.reset_default_graph()
#         X_batch = tf.placeholder(tf.float32, shape=[None, 126])
#         y = self.ann(X_batch, training=False)
#         saver = tf.train.Saver()
#         # predict
#         with tf.Session() as sess:
#             saver.restore(sess, self.path)
#             _y = sess.run(y, feed_dict={X_batch: _X_batch})
#         _y_inverse = self.y_pipeline.inverse_transform(_y)
#         return _y
