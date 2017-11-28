#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import dill as pickle
import IPython
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import itertools
import math
import types
import copy
from collections import OrderedDict

# scipy
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer, StandardScaler
from shared import LabelBinarizerPipelineFriendly
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import linear_model

# pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

import shared

def udf_nn(*args):
    layers = []
    for i in range(len(args)-1):
        layers.append(nn.Linear(args[i], args[i+1]))
        if i != len(args)-2:
            layers.append(nn.ELU())
    return Sequential(*layers)

# save, load
# ==============================================================================

def save(obj, middlename):      # Note! Where defined, above which obj pickled.
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


# MlVaspSpeed
# ==============================================================================


class MlVaspSpeedNet(nn.Module):

    def __init__(self, bn_momentum=0.97, dropout_p=0.077):
        super(MlVaspSpeedNet, self).__init__()
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

        self.lC1 = nn.Linear(10, 2)
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

        C = self.bnC1(self.dropoutC1(F.elu(self.lC1(X[:, 8:]))))
        C = self.lC2(C)
        # C = self.lC3(X[:, 8:12])

        y = torch.cat((A, B, C), dim=1)
        y = self.bn0(self.dropout0(F.elu(y)))
        y = self.bn1(self.dropout1(F.elu(self.l1(y))))
        y = self.l2(y)
        # y = A * B * C

        return y


class MlVaspSpeed(object):

    def __init__(self):
        # data
        self._X = []
        self._y0 = []
        self._cur = []
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
                    ('slice_flatten', FunctionTransformer(func=lambda X: X[:, 8:9])),
                    ('labeler', LabelBinarizerPipelineFriendly()),
                    ('padder', FunctionTransformer(func=lambda X: np.hstack((X, np.zeros((X.shape[0], 10-X.shape[1]))))))
                ]))
            ])),
            ('cast_to_float32', FunctionTransformer(func=np.float32))
        ])
        self.y_pipeline = Pipeline([
            ('log', FunctionTransformer(func=np.log, inverse_func=np.exp)),      # reduce information to reasonable
            ('scaler', StandardScaler())
        ])
        # ann. what a pity.
        self.net = MlVaspSpeedNet()


    def parse_train(self, node, vasp, gen, cell, makeparam):
        # OUTPUT
        _y0 = [vasp.info('run_time') / vasp.info('n_electronic_step')]
        # INPUT
        # -----
        _X = [
            makeparam.projector_real + makeparam.projector_reciprocal,
            makeparam.wavefunction,
            makeparam.arraygrid,
            cell.natoms(),
            np.dot(np.cross(cell.base[0], cell.base[1]), cell.base[2]),
            # platform-specific
            int(gen.getkw('ncore_node')),
            int(gen.getkw('nnode')),
            int(gen.getkw('ncore')),
            gen.getkw('platform')
        ]
        # COMMENT (posterity)
        # ------------------
        _cur = node.default_path(cur=True)
        # put it here so that no inconsistency will happen
        self._X.append(_X)
        self._y0.append(_y0)
        self._cur.append(_cur)
        return (_X, _y0, _cur)


    def train(self, n_epochs=5800, batch_size=64, learning_rate=0.026, optimizer_name='SGD', test_set_size=5):
        test_idx = np.random.choice(range(len(self._X)), size=test_set_size)
        train_idx = np.array([i for i in range(len(self._X)) if i not in test_idx])

        # train
        # pipeline
        _X = self.X_pipeline.fit_transform(self._X)[train_idx]
        _y0 = self.y_pipeline.fit_transform(self._y0)[train_idx]
        # batch: random.choice
        # ann
        criterion = nn.MSELoss()
        optimizer = getattr(optim, optimizer_name)(self.net.parameters(), lr=learning_rate)
        # train
        self.net.train()
        for epoch in range(n_epochs):
            batch_idx = np.random.choice(range(_X.shape[0]), size=batch_size)
            X_batch= Variable(torch.FloatTensor(_X[batch_idx]), requires_grad=True)
            y0_batch = Variable(torch.FloatTensor(_y0[batch_idx]), requires_grad=False)
            y = self.net(X_batch)
            loss = criterion(y, y0_batch)
            optimizer.zero_grad()   # suggested trick
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print 'epoch %s, loss %s'%(epoch, loss.data.numpy()[0])

        # test
        _X = np.array(self._X)[test_idx]
        _y0 = np.float32(self._y0).flatten()[test_idx]
        _y = np.float32(self.predict(_X)).flatten()
        print self.__class__.__name__ + '.train: training finished. evaluation on last items: \n actual | predicted'
        for a, b in zip(_y0, _y):
            print a, b
        a = np.zeros((_y0.shape[0], 2))
        a[:, 0] = _y0 / _y
        a[:, 1] = _y / _y0
        b = np.amax(a, axis=0)
        return np.sum(b ** 2.5) / 100   # yes, I'm using a different loss. the point, however, is that I don't want to blow up the convergence.


    def parse_predict(self, gen, cell, makeparam):
        return [[
            makeparam.projector_real + makeparam.projector_reciprocal,
            makeparam.wavefunction,
            makeparam.arraygrid,
            cell.natoms(),
            np.dot(np.cross(cell.base[0], cell.base[1]), cell.base[2]),
            # platform-specific
            int(gen.getkw('ncore_node')),
            int(gen.getkw('nnode')),
            int(gen.getkw('ncore')),
            gen.getkw('platform')
        ]]

    def predict(self, _X):
        # pipeline
        _X = self.X_pipeline.transform(_X)
        # ann
        self.net.eval()
        y = self.net(Variable(torch.FloatTensor(_X), requires_grad=True))
        # pipeline
        _y_inverse = self.y_pipeline.inverse_transform(y.data.cpu().numpy())
        return _y_inverse


# inital training script for MlVaspSpeed can be found in scripts/mly



# Ml
# ==============================================================================
class MlQueueTime(object):

    def __init__(self):
        # data
        self._X = []
        self._y0 = []
        self._cur = []
        # pipeline
        self.X_pipeline = Pipeline([
            ('cast_to_array', FunctionTransformer(func=np.array)),
            ('split', FeatureUnion(transformer_list=[
                ('nnode', Pipeline([
                    ('slicer', FunctionTransformer(func=lambda X: X[:, :-1])),
                ])),
                ('platform', Pipeline([
                    ('slice_flatten', FunctionTransformer(func=lambda X: X[:, -1:])),
                    ('labeler', LabelBinarizerPipelineFriendly()),
                    ('padder', FunctionTransformer(func=lambda X: np.hstack((X, np.zeros((X.shape[0], 10-X.shape[1]))))))
                ]))
            ])),
            ('scaler', StandardScaler())
        ])
        self.y_pipeline = Pipeline([
            ('log', FunctionTransformer(func=np.log, inverse_func=np.exp)),      # reduce information to reasonable
            ('scaler', StandardScaler())
        ])
        # ann. what a pity.
        self.net = udf_nn(11, 5, 5, 1)


    def parse_train(self, node, vasp, gen):
        # OUTPUT
        _y0 = [vasp.info('queue_time')]
        # INPUT
        # -----
        _X = [
            int(gen.getkw('nnode')),
            gen.getkw('platform')
        ]
        # COMMENT (posterity)
        # ------------------
        _cur = node.default_path(cur=True)
        # put it here so that no inconsistency will happen
        self._X.append(_X)
        self._y0.append(_y0)
        self._cur.append(_cur)
        return (_X, _y0, _cur)


    def train(self, n_epochs=5800, batch_size=64, learning_rate=0.026, optimizer_name='SGD', test_set_size=5):
        test_idx = np.random.choice(range(len(self._X)), size=test_set_size)
        train_idx = np.array([i for i in range(len(self._X)) if i not in test_idx])

        # train
        # pipeline
        _X = self.X_pipeline.fit_transform(self._X)[train_idx]
        _y0 = self.y_pipeline.fit_transform(self._y0)[train_idx]
        # batch: random.choice
        # ann
        criterion = nn.MSELoss()
        optimizer = getattr(optim, optimizer_name)(self.net.parameters(), lr=learning_rate)
        # train
        self.net.train()
        for epoch in range(n_epochs):
            batch_idx = np.random.choice(range(_X.shape[0]), size=batch_size)
            X_batch= Variable(torch.FloatTensor(_X[batch_idx]), requires_grad=True)
            y0_batch = Variable(torch.FloatTensor(_y0[batch_idx]), requires_grad=False)
            y = self.net(X_batch)
            loss = criterion(y, y0_batch)
            optimizer.zero_grad()   # suggested trick
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print 'epoch %s, loss %s'%(epoch, loss.data.numpy()[0])

        # test
        _X = np.array(self._X)[test_idx]
        _y0 = np.float32(self._y0).flatten()[test_idx]
        _y = np.float32(self.predict(_X)).flatten()
        print self.__class__.__name__ + '.train: training finished. evaluation on last items: \n actual | predicted'
        for a, b in zip(_y0, _y):
            print a, b
        print self.__class__.__name__ + ': a full queuetime prediction would be time-wasting and messy. This is just an averge, good enough for slow times not not necessary for fast times. :)'


    def parse_predict(self, gen):
        return [[
            int(gen.getkw('ncore_node')),
            gen.getkw('platform')
        ]]

    def predict(self, _X):
        # pipeline
        _X = self.X_pipeline.transform(_X)
        # ann
        self.net.eval()
        y = self.net(Variable(torch.FloatTensor(_X), requires_grad=True))
        # pipeline
        _y_inverse = self.y_pipeline.inverse_transform(y.data.numpy())
        return _y_inverse



# MlPBSOpt_X_CE1_DiscreteLC
# =========================

def distdict(N):
    dist = np.unique([round(np.linalg.norm([i,j,k])*sgn, 4) for i in range(N) for j in range(N) for k in range(N) for sgn in (-1,1)])
    ddict = OrderedDict(zip(dist, np.zeros(dist.shape[0])))
    return ddict

class MlPbSOptXC1D(object):

    def __init__(self):
        # data
        self._X = []
        self._y0 = []
        # pipeline
        # ann
        self.model = linear_model.LinearRegression()

    def parse_X(self, cell):
        '''
        ccoor is coordinates. natom0 is # of Pb atoms (for determining sgn).
        returns a list.
        '''
        ccoor = cell.ccoor
        natom0 = cell.stoichiometry.values()[0]
        X = []
        for i, ci in enumerate(ccoor):
            for ix in range(3):
                feature = distdict(8)
                for j, cj in enumerate(ccoor):
                    if j==i: continue
                    x = np.around((cj-ci)/3.007)
                    r = np.linalg.norm(x)
                    sgn = np.sign((i - natom0 + 0.5) * (j - natom0 + 0.5))
                    feature[round(sgn*r, 4)] += x[ix] / r
                X.append(feature.values())
        return X

    def parse_y0(self, vasp):
        ccoor = vasp.node().cell.ccoor
        dcoor = vasp.optimized_cell.ccoor - vasp.node().cell.ccoor
        X = []
        for i, ci in enumerate(ccoor):
            #
            dc = dcoor[i]
            #
            neighbor = [j for j,cj in enumerate(ccoor) if j!=i and np.linalg.norm(cj-ci)<4]
            neighbor_avg_dc = np.mean(dcoor[neighbor], axis=0)
            dc -= neighbor_avg_dc
            #
            for ix in range(3):
                X.append(dc[ix])
        return X

    def parse_train(self, vasp):
        '''More of a handle.'''
        self._X += list(self.parse_X(vasp.node().cell))
        self._y0 += list(self.parse_y0(vasp))

    def train(self, test_set_size=10, total_set_size=None):
        # pipeline
        total_idx = np.random.choice(range(len(self._y0)), size=total_set_size if total_set_size else len(self._y0), replace=False)
        test_idx = np.random.choice(total_idx, size=test_set_size, replace=False)
        train_idx = np.array([i for i in total_idx if i not in test_idx])
        _X = np.array(self._X)[train_idx]
        _y0 = np.array(self._y0)[train_idx]
        model = self.model
        # train
        model.fit(_X, _y0)

        # evaluate
        # train
        _X_train = np.array(self._X)[train_idx]
        _y0_train = np.array(self._y0)[train_idx]
        _y_train = self.predict(_X_train)
        # test
        _X_test = np.array(self._X)[test_idx]
        _y0_test = np.array(self._y0)[test_idx]
        _y_test = self.predict(_X_test)
        IPython.embed()

    def parse_predict(self, cell):
        return self.parse_X(cell)

    def predict(self, _X):
        return self.model.predict(_X)


# MlPbSOptXRNN
# ==============================================================================

from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking
from keras.preprocessing.sequence import pad_sequences

class MlPbSOptXRNN(object):

    def __init__(self):
        # data
        self._X = []
        self._y0 = []
        # pipeline
        self.X_pipeline = StandardScaler(with_mean=False)
        self.y_pipeline = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('15', FunctionTransformer(func=lambda x: x * 15, inverse_func=lambda x: x / 15))
        ])
        # ann
        self.timesteps = 600
        self.data_dim = 4
        self.model = Sequential([
            Masking(mask_value=0.0, input_shape=(self.timesteps, self.data_dim)),
            LSTM(8, return_sequences=True),
            LSTM(8),
            Dense(8, activation='relu'),
            Dense(24, activation='relu'),
            Dense(3)
        ])
        self.model.compile(loss='mean_absolute_percentage_error',
                      optimizer='adam',
                      metrics=['accuracy'])


    def parse_X(self, cell):
        '''
        ccoor is coordinates. natom0 is # of Pb atoms (for determining sgn).
        returns a list.
        '''
        ccoor = cell.ccoor
        natom0 = cell.stoichiometry.values()[0]
        X1 = []
        for i, c in enumerate(ccoor):
            dcoor = ccoor - c
            sgn = np.sign((i - natom0 + 0.5) * (np.arange(len(ccoor)) - natom0 + 0.5))
            dcoorp = np.concatenate((dcoor, np.c_[sgn]), axis=1)
            X1.append(np.delete(dcoorp, i, axis=0))
        return X1

    def parse_y0(self, vasp):
        return vasp.optimized_cell.ccoor - vasp.node().cell.ccoor

    def parse_train(self, vasp):
        '''More of a handle.'''
        self._X += list(self.parse_X(vasp.node().cell))
        self._y0 += list(self.parse_y0(vasp))

    def train(self):
        # pipeline
        _X = self.X_pipeline.fit_transform(pad_sequences(self._X, dtype='float32', maxlen=self.timesteps).reshape(-1, self.data_dim)).reshape(-1, self.timesteps, self.data_dim)
        _y0 = self.y_pipeline.fit_transform(self._y0)
        model = self.model
        # fit
        model.fit(_X, _y0, batch_size=64, epochs=500)
        _y = model.predict(_X)
        IPython.embed()
