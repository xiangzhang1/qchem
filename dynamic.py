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
        _y_inverse = self.y_pipeline.inverse_transform(y.data.numpy())
        return _y_inverse


# inital training script for MlVaspSpeed can be found in scripts/ml


# MlPbSOpt, f(r)-version
# ==============================================================================


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


class MlPbSOpt(object):
    '''Deprecated.'''

    def __init__(self):
        # data
        self._X = []
        self._y0 = []
        # pipeline
        self.X_pipeline = MlPbSOptScaler()
        self.y_pipeline = Pipeline([
            ('scaler', MlPbSOptScaler()),
            ('10', FunctionTransformer(func=lambda x: x * 5, inverse_func=lambda x: x / 5))
        ])
        # ann
        self.nets = {-2: udf_nn(1,5,10,5,1), 0:udf_nn(1,5,10,5,1), 2:udf_nn(1,5,10,5,1)}

    def parse_train(self, vasp):
        # checks
        gen = vasp.node().gen
        vasprunxml_lastline = vasp.ssh_and_run('tail -1 %s/vasprun.xml' %vasp.remote_folder_name).splitlines()[0]
        if not (gen.parse_if('opt') and vasp.info('n_ionic_step') < int(gen.getkw('nsw')) and '</modeling>' in vasprunxml_lastline):
            raise shared.CustomError('{}.compute: not optimization cell, or terminated prematurely. skipped :)'.format(self.__class__.__name__))

        a = 6.01417/2
        cell = vasp.optimized_cell

        ccoor = np.copy(cell.ccoor)
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



    def train(self, n_epochs=400, learning_rate=0.001, optimizer_name='Adam'):
        # pipeline
        _X = copy.deepcopy(self._X)
        self.X_pipeline.fit(np.concatenate([_subX[:,:3] for _subX in _X], axis=0))
        for i in range(len(_X)):
            _X[i][:,:3] = self.X_pipeline.transform(_X[i][:,:3])
        _y0 = self.y_pipeline.fit_transform(self._y0)
        # batch
        # ann
        criterion = nn.MSELoss()
        params = list(self.nets[-2].parameters()) + list(self.nets[0].parameters()) + list(self.nets[2].parameters())
        optimizer = getattr(optim, optimizer_name)(params, lr=learning_rate)
        # train
        [net.train() for net in self.nets.values()]
        for epoch in range(n_epochs):
            for _X_batch, _y0_batch in zip(_X[:-50], _y0[:-50]):
                dx = Variable(torch.zeros(3))
                for sgn in [-2,0,2]:
                    indices, = np.where([row[3] + row[4] == sgn for row in _X_batch])
                    if not len(indices): continue
                    X = Variable(torch.FloatTensor(_X_batch[indices]))

                    r = torch.norm(X[:, :3], p=2, dim=1, keepdim=True)      # (N,3) -> (N,1)
                    rhat = X[:, :3] / r * X[:, 3:4] * X[:, 4:5]     # (N,3) / (N,1)
                    dxi = self.nets[sgn](r) * rhat     # (N,1) * (N,1) * (N,1) * (N,3)
                    dx += torch.sum(dxi, dim=0, keepdim=False)    # (N,3) -> (3)

                dx0 = Variable(torch.FloatTensor(_y0_batch))
                loss = criterion(dx, dx0)
                optimizer.zero_grad()   # suggested trick
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print 'epoch %s, loss %s' %(epoch, np.asscalar(loss.data.numpy()))

        # test
        print self.__class__.__name__ + '.train: training finished. evaluation on last items: \n actual | error'
        for i in range(len(self._X)-50, len(self._X)):
            _X = np.array(self._X)[i]
            _y0 = np.float32(self._y0)[i]
            _y = np.float32(self.predict(_X))[0]
            print _y0, _y

    def parse_predict(self, cell): # cell -> (Natom, xyzs)
        a = 6.01417/2
        ccoor = cell.ccoor
        # parse and store
        features = []
        pbs_order_factor = 1 if cell.stoichiometry.keys()[0]=='Pb' else -1
        for idx_atom, c in enumerate(ccoor):
            relative_ccoor = ccoor - c
            coor_sgn = np.sign(np.arange(ccoor.shape[0]) - cell.stoichiometry.values()[0] + 0.5) * pbs_order_factor
            sgn = np.sign(idx_atom - cell.stoichiometry.values()[0] + 0.5) * pbs_order_factor
            feature = np.concatenate((relative_ccoor, np.c_[[sgn] * ccoor.shape[0]], np.c_[coor_sgn]), axis=1)
            feature = np.delete(feature, idx_atom, 0)
            features.append(feature)
        return features

    def predict(self, _X):  # (Natom, xyzs) -> (1, Natom)
        # pipeline
        _X = copy.deepcopy(_X)
        _X[:,:3] = self.X_pipeline.transform(_X[:,:3])
        # ann
        [net.eval() for net in self.nets.values()]

        dx = Variable(torch.zeros(3))
        for sgn in [-2,0,2]:
            indices, = np.where([row[3] + row[4] == sgn for row in _X])
            if not len(indices): continue
            X = Variable(torch.FloatTensor(_X[indices]))

            r = torch.norm(X[:, :3], p=2, dim=1, keepdim=True)      # (N,3) -> (N,1)
            rhat = X[:, :3] / r * X[:, 3:4] * X[:, 4:5]     # (N,3) / (N,1)
            dxi = self.nets[sgn](r) * rhat     # (N,1) * (N,1) * (N,1) * (N,3)
            dx += torch.sum(dxi, dim=0, keepdim=False)    # (N,3) -> (3)

        # pipeline
        _y_inverse = self.y_pipeline.inverse_transform(dx.data.numpy().reshape(1,-1)).reshape(-1)   # Scaler requries 2D array.
        return _y_inverse


    def optimize(self, cell):
        #
        cell = copy.deepcopy(cell)
        Xs = self.parse_predict(cell)
        #
        for idx_atom in range(cell.natoms()):
            X = Xs[idx_atom]
            dx = self.predict(X)
            cell.ccoor[idx_atom] += dx
        return cell




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



# MlPbSOpt, Force-CE-version
# ==============================================================================

def V(x):
    return Variable(torch.FloatTensor(np.array(x)), requires_grad=True)

class MlPbSOptFCE(object):

    def __init__(self):
        # data
        self._X1 = []
        self._y0 = []
        # pipeline
        self.X1_pipeline = StandardScaler(with_mean=False)
        self.y_pipelien = StandardScaler(with_mean=False)
        # ann
        self.ce1 = udf_nn(4, 20, 20, 1)

    def parse_X1(self, ccoor, natom0):
        '''
        ccoor is coordinates. natom0 is # of Pb atoms (for determining sgn).
        returns a list.
        '''
        X1 = []
        for i, c in enumerate(ccoor):
            dcoor = ccoor - c
            sgn = np.sign((i - natom0 + 0.5) * (np.arange(len(ccoor)) - natom0 + 0.5))
            dcoorp = np.concatenate((dcoor, np.c_[sgn]), axis=1)
            dcoorp.pop(i)
            X1.append(dcoorp)
        return X1

    def parse_y0(self, vasp):
        cell = vasp.optimized_cell
        os.chdir(vasp.node().path)
        with open('OUTCAR', 'r') as f:
            lines = f.readlines()
        i = next(i for i, line in enumerate(lines) if 'TOTAL-FORCE' in line)
        force_lines = lines[i+2: i+2+cell.natoms()]
        forces = np.float32([line.split() for line in force_lines])[:, 3:]
        return forces

    def parse_train(self, vasp):
        '''More of a handle.'''
        self._X1 += self.parse_X1(vasp.node().cell.ccoor, vasp.node().cell.stoichiometry[0])
        self._y0 += self.parse_y0(vasp)

    def train(self, n_epochs=400, learning_rate=0.001, optimizer_name='Adam'):
        # pipeline
        self.X1_pipeline.fit(np.concatenate(self._X1, axis=0))
        _X1 = np.array([self.X1_pipeline.transform(_X1_) for _X1_ in self._X1])
        _y0 = self.y_pipelien.fit_transform(self._y0)
        ce1 = self.ce1
        # batch
        # ann
        criterion = nn.MSELoss()
        params = list(ce1.parameters())
        optimizer = getattr(optim, optimizer_name)(params, lr=learning_rate)
        # train
        ce1.train()
        for epoch in range(n_epochs * len(_X1)):
            i = np.random.randint(0, len(_X1) - 50)
            X1 = V(_X1[i])
            f0 = V(_y0[i])

            origin = V([0,0,0])
            X1_[:,:3] = X1_[:,:3] - origin
            e = torch.sum(ce1(X1_), keepdim=False)
            f = torch.autograd.grad(e, origin, create_graph=True)

            loss = criterion(f, f0_)
            optimizer.zero_grad()   # suggested trick
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print 'epoch %s, loss %s' %(epoch, np.asscalar(loss.data.numpy()))

    def parse_predict(self, ccoor, natom0):
        return self.parse_X1(ccoor, natom0)

    def predict(self, _X1):
        # pipeline
        _X1 = self.X_pipeline.transform(_X)
        ce1 = self.ce1
        # ann
        ce1.eval()
        y = torch.sum(ce1(V(_X1)), keepdim=True)
        # reverse pipeline
        return np.asscalar(self.y_pipeline.inverse_transform(y.data.numpy()))
