#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import dill as pickle
import IPython
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import math
import types
import copy

# scipy
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

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
from torch.nn import Sequential
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

import shared

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
        self.net = MlVaspSpeedNet()


    def parse_train(self, node, vasp, gen, cell, makeparam):
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
                raise shared.CustomError(self.__class__.__name__ + '.warning: total time does not feel right. skipped collect data.')
            # number of ionic steps
            iteration_lines = [l for l in lines if 'Iteration' in l]
            number_elec_steps = len(iteration_lines)
            # time per electronic step
            time_elec_step = total_time / number_elec_steps
        _y0 = [time_elec_step]
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
        _y_inverse = self.y_pipeline.inverse_transform(y.data.numpy())
        return _y_inverse


# inital training script for MlVaspSpeed can be found in scripts/pytorch_learn


# MlPbSOpt
# ==============================================================================

from sklearn.base import BaseEstimator, TransformerMixin
class MlPbSOptScaler(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.mean = 0

    def fit(self, X):
        self.mean = np.mean([np.mean(np.abs(subX)) for subX in X])
        return self

    def transform(self, X):
        return X / self.mean / 1.7

    def inverse_transform(self, X):
        return X * self.mean * 1.7

def udf_nn(*args):
    layers = []
    for i in range(len(args)-1):
        layers.append(nn.Linear(args[i], args[i+1]))
        layers.append(nn.ELU())
    return Sequential(*layers)

class MlPbSOpt(object):

    def __init__(self):
        # data
        self._X = []
        self._y0 = []

        # pipeline
        # method 1
        self.X_pipeline = MlPbSOptScaler()
        self.y_pipeline = MlPbSOptScaler()
        # # method 2
        # self.X_pipeline = StandardScaler()
        # self.y_pipeline = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('_10', FunctionTransformer(func=lambda x: x * 10, inverse_func=lambda x: x / 10))
        # ])

        # ann
        self.nets = {-2: udf_nn(1,5,10,5,1), 0:udf_nn(1,5,10,5,1), 2:udf_nn(1,5,10,5,1)}

    def parse_train(self, vasp):
        a = 6.01417/2
        cell = vasp.optimized_cell

        # snap to grid
        ccoor = cell.ccoor
        origin = ccoor[cell.ccoor_kdtree().query(np.mean(ccoor, axis=0))[1]]    # closest-to-center atom
        def error_after_transformation(origin, ccoor=ccoor, a=a):       # snap to grid. note: parallelization doesn't save time.
            fcoor = (ccoor - origin) / a
            return np.sum(np.abs(fcoor - np.around(fcoor)))
        origin = minimize(fun=error_after_transformation,
                     x0=origin,
                     bounds=[(origin[i]-0.2*a, origin[i]+0.2*a) for i in range(3)],
                     tol=1e-10
                    ).x
        ccoor = ccoor - origin + np.around(origin / a) * a      # on-grid coordinate. vaguely resemble the original cell

        # parse and store
        pbs_order_factor = 1 if cell.stoichiometry.keys()[0]=='Pb' else -1
        for idx_atom, c in enumerate(ccoor):
            relative_ccoor = ccoor - c
            coor_sgn = np.sign(np.arange(ccoor.shape[0]) - cell.stoichiometry.values()[0] + 0.5) * pbs_order_factor
            sgn = np.sign(idx_atom - cell.stoichiometry.values()[0] + 0.5) * pbs_order_factor
            feature = np.concatenate((relative_ccoor, np.c_[[sgn] * ccoor.shape[0]], np.c_[coor_sgn]), axis=1)
            feature = np.delete(feature, idx_atom, 0)
            label = c - np.around(c/a)*a
            self._X.append(feature)
            self._y0.append(label)


    def train(self, n_epochs=100, learning_rate=0.001, optimizer_name='Adam'):

        # train
        # pipeline
        _X = copy.deepcopy(self._X)
        self.X_pipeline.fit(np.concatenate([_subX[:,:3] for _subX in _X], axis=0))
        for i in range(len(_X)):
            _X[i][:,:3] = self.X_pipeline.transform(_X[i][:,:3])
        _y0 = self.y_pipeline.fit_transform(self._y0)
        # batch: random.choice
        # ann
        criterion = nn.MSELoss()
        IPython.embed()
        optimizer = getattr(optim, optimizer_name)(itertools.chain(net.parameters() for net in self.nets.values()), lr=learning_rate)
        # train
        [net.train() for net in self.nets.values()]
        for epoch in range(n_epochs):
            for _X_batch, _y0_batch in zip(_X[:-50], _y0[:-50]):

                # method 3
                dx = Variable(torch.zeros(3))
                for sgn in [-2,0,2]:
                    indices = np.where([row[3] + row[4] == sgn for row in _X_batch])
                    if not indices: continue

                    X = Variable(torch.FloatTensor(_X_batch[indices]))

                    # # method 1
                    # r = torch.norm(X[:, :3], p=2, dim=1, keepdim=True)      # (N,3) -> (N,1)
                    # rhat = X[:, :3] / r     # (N,3) / (N,1)
                    # dx = self.net(r) * X[:, 3:4] * X[:, 4:5] * rhat     # (N,1) * (N,1) * (N,1) * (N,3)
                    # dx = torch.sum(dx, dim=0, keepdim=False)    # (N,3) -> (3)
                    # # method 2
                    # dx = self.net(X)    #(N,3) * (N,1) * (N,1)
                    # dx = torch.sum(dx, dim=0, keepdim=False)    # (N,3) -> (3)
                    # method 3
                    r = torch.norm(X[:, :3], p=2, dim=1, keepdim=True)      # (N,3) -> (N,1)
                    rhat = X[:, :3] / r     # (N,3) / (N,1)
                    dx = self.nets[sgn](r) * rhat     # (N,1) * (N,1) * (N,1) * (N,3)
                    dx += torch.sum(dx, dim=0, keepdim=False)    # (N,3) -> (3)

            dx0 = Variable(torch.FloatTensor(_y0_batch))
            loss = criterion(dx, dx0)
            optimizer.zero_grad()   # suggested trick
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print 'epoch %s, loss %s' %(epoch, np.asscalar(loss.data.numpy()))

        # test
        print self.__class__.__name__ + '.train: training finished. evaluation on last items: \n actual | predicted'
        for i in range(len(self._X)-50, len(self._X)):
            _X = np.array(self._X)[i]
            _y0 = np.float32(self._y0)[i]
            _y = np.float32(self.predict(_X))
            print _y0, _y


    def parse_predict(self, gen, cell, makeparam):
        pass

    def predict(self, _X):
        # pipeline
        _X = copy.deepcopy(_X)
        _X[:,:3] = self.X_pipeline.transform(_X[:,:3])
        # ann
        self.net.eval()
        X = Variable(torch.FloatTensor(_X))

        # method 1
        r = torch.norm(X[:, :3], p=2, dim=1, keepdim=True)      # (N,3) -> (N,1)
        rhat = X[:, :3] / r     # (N,3) / (N,1)
        dx = self.net(r) * X[:, 3:4] * X[:, 4:5] * rhat     # (N,1) * (N,1) * (N,1) * (N,3)
        # # method 2
        # dx = self.net(X)    #(N,3) * (N,1) * (N,1)

        dx = torch.sum(dx, dim=0, keepdim=False)    # (N,3) -> (3)
        # pipeline
        _y_inverse = self.y_pipeline.inverse_transform(dx.data.numpy().reshape(1,-1))
        return _y_inverse
