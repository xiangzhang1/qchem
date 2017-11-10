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

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class MlPbSOpt(object):

    def __init__(self):
        # data
        self._X_local = []
        self._X_high = []
        self._X_global = []
        self._y0 = []
        # pipeline
        self.X_high_pipeline = StandardScaler()
        self.y_pipeline = StandardScaler()
        # ann. have fun!
        bn_momentum = 0.74
        dropout_p = 0.001
        self.net_local = Sequential(
            nn.Linear(125, 25),
            nn.BatchNorm1d(25, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(25, 20),
            nn.BatchNorm1d(20, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(20, 15),
            nn.BatchNorm1d(15, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(15, 10),
            nn.BatchNorm1d(10, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(10, 5),
            nn.BatchNorm1d(5, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
        ).cuda()
        self.net_high = Sequential(
            nn.Linear(9, 8),
            nn.BatchNorm1d(8, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(8, 7),
            nn.BatchNorm1d(7, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(7, 6),
            nn.BatchNorm1d(6, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
        ).cuda()
        self.net_global = Sequential(
            nn.Conv3d(2, 2, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(2, 2, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool3d(2),
            Reshape(-1, 2 * 2 * 2 * 2),
            nn.Linear(2 * 2 * 2 * 2, 16),
            nn.BatchNorm1d(16, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(16, 12),
            nn.BatchNorm1d(12, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(12, 8),
            nn.BatchNorm1d(8, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4, momentum=bn_momentum),
            nn.Dropout(p=dropout_p)
        ).cuda()
        self.net_final = Sequential(
            nn.Linear(15, 12),
            nn.BatchNorm1d(12, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(12, 8),
            nn.BatchNorm1d(8, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4, momentum=bn_momentum),
            nn.Dropout(p=dropout_p),
            nn.Linear(4, 1)
        ).cuda()


    def parse_train(self, vasp):
        a = 6.01417/2
        cell = vasp.optimized_cell

        # 三一. 确定初始格子
        ccoor = cell.ccoor
        def error_after_transformation(transformation, ccoor=ccoor, a=a):       # note: parallelization doesn't save time.
            x0, y0, z0, sz, sx, sy = transformation
            rotation_matrix = shared.euler2mat(sz, sx, sy)
            ccoor_transformed = np.dot(np.subtract(ccoor, [x0, y0, z0]), rotation_matrix.T)
            fcoor = ccoor_transformed / a
            return np.linalg.norm(fcoor - np.around(fcoor))
        transformation = minimize(fun=error_after_transformation,
                     x0=[0, 0, 0, 0, 0, 0],
                     bounds=[(-.8,.8), (-.8,.8), (-.8,.8), (-.2,.2), (-.2,.2), (-.2,.2)]
                    ).x
        x0, y0, z0, sz, sx, sy = transformation
        rotation_matrix = shared.euler2mat(sz, sx, sy)
        ccoor_transformed = np.dot(np.subtract(ccoor, [x0, y0, z0]), rotation_matrix.T)
        fcoor = ccoor_transformed / a
        rfcoor = np.around(fcoor)

        # 二. 将稀疏网格转为密集网格
        Nx = int(math.ceil(np.linalg.norm(cell.base[0]) / a))      # assume 长方体
        Ny = int(math.ceil(np.linalg.norm(cell.base[1]) / a))
        Nz = int(math.ceil(np.linalg.norm(cell.base[2]) / a))
        dense_matrix = np.zeros((Nx, Ny, Nz, 4))
        for idx_atom, rfc in enumerate(rfcoor):
            # 坐标
            ix, iy, iz = np.int32(rfc)
            # 符号位
            idx_ele = 0 if idx_atom < cell.stoichiometry.values()[0] else 1     # OrderedDict顺序     # assume Pb S only
            symbol = cell.stoichiometry.keys()[idx_ele]
            feature_n = 1 if symbol=='Pb' else -1
            dense_matrix[ix, iy, iz, 0] = feature_n
            # 数值位: dx, dy, dz
            dense_matrix[ix, iy, iz, 1:] = fcoor[idx_atom] - rfc

        feature_stoichiometry = np.float32([cell.stoichiometry['Pb'], cell.stoichiometry['S']])

        # 五. 对称性
        dense_matrices  = [dense_matrix[::reverse_x, ::reverse_y, ::reverse_z, :].transpose(order) for reverse_x in [-1,1] for reverse_y in [-1,1] for reverse_z in [-1,1] for order in [(0,1,2,3),(0,2,1,3),(1,0,2,3),(1,2,0,3),(2,1,0,3),(2,0,1,3)]]
        for dense_matrix in dense_matrices[0:2]:
            center_coordinate = np.mean([[ix,iy,iz] for ix,iy,iz in np.ndindex((Nx,Ny,Nz)) if dense_matrix[ix,iy,iz,0]!=0], axis=0)

            for ix, iy, iz in np.ndindex((Nx,Ny,Nz)):
                if dense_matrix[ix,iy,iz,0] != 0:
                    # 三二. 取得 dx 和 local feature
                    feature_local = dense_matrix[ix-2:ix+3, iy-2:iy+3, iz-2:iz+3, 0].flatten()    # C式拍平，质量保证！
                    label_dx = dense_matrix[ix,iy,iz,1:2]

                    feature_global = np.pad(dense_matrix, 5, mode='constant')[ix:ix+11, iy:iy+11, iz:iz+11, 0:2]
                    feature_global[:,:,:,1] = feature_global[:,:,:,0]**2
                    feature_global = feature_global.transpose((3,0,1,2))

                    # 四. 关于高级策略
                    feature_selfcharge = dense_matrix[ix, iy, iz, 0:1]
                    feature_local *= feature_selfcharge[0]
                    feature_global *= feature_selfcharge[0]
                    feature_displace_to_center = np.float32([ix,iy,iz]) - center_coordinate

                    nsd1 = next(k for k,g in enumerate(dense_matrix[ix:,iy,iz,0]) if g==0)
                    nsd2 = next(k for k,g in enumerate(dense_matrix[ix:0:-1,iy,iz,0]) if g==0)
                    fdtsx = min(nsd1, nsd2)
                    nsd1 = next(k for k,g in enumerate(dense_matrix[ix,iy:,iz,0]) if g==0)
                    nsd2 = next(k for k,g in enumerate(dense_matrix[ix,iy:0:-1,iz,0]) if g==0)
                    fdtsy = min(nsd1, nsd2)
                    nsd1 = next(k for k,g in enumerate(dense_matrix[ix,iy,iz:,0]) if g==0)
                    nsd2 = next(k for k,g in enumerate(dense_matrix[ix,iy,iz:0:-1,0]) if g==0)
                    fdtsz = min(nsd1, nsd2)

                    self._X_local.append(feature_local)
                    self._X_high.append(np.concatenate((feature_stoichiometry, feature_selfcharge, feature_displace_to_center, [fdtsx, fdtsy, fdtsz])))
                    self._X_global.append(feature_global)
                    self._y0.append(label_dx)


    def train(self, n_epochs=8000, batch_size=64, learning_rate=0.001, optimizer_name='SGD', test_set_size=128):
        test_idx = np.random.choice(range(len(self._X_local)), size=test_set_size)
        train_idx = np.array([i for i in range(len(self._X_local)) if i not in test_idx])

        # train
        # pipeline
        _X_local = np.float32(self._X_local)[train_idx]
        _X_high = self.X_high_pipeline.fit_transform(self._X_high)[train_idx]
        _X_global = np.float32(self._X_global)[train_idx]
        _y0 = self.y_pipeline.fit_transform(self._y0)[train_idx]
        # batch: random.choice
        # ann
        criterion = nn.MSELoss()
        optimizer = getattr(optim, optimizer_name)(itertools.chain(self.net_local.parameters(), self.net_high.parameters(), self.net_global.parameters(), self.net_final.parameters()), lr=learning_rate)
        # train
        self.net_local.train()
        self.net_high.train()
        self.net_global.train()
        self.net_final.train()
        for epoch in range(n_epochs):
            batch_idx = np.random.choice(range(_X_local.shape[0]), size=batch_size)
            #
            X_local= Variable(torch.FloatTensor(_X_local[batch_idx]), requires_grad=True).cuda()
            X_high= Variable(torch.FloatTensor(_X_high[batch_idx]), requires_grad=True).cuda()
            X_global= Variable(torch.FloatTensor(_X_global[batch_idx]), requires_grad=True).cuda()
            y0 = Variable(torch.FloatTensor(_y0[batch_idx]), requires_grad=False).cuda()
            #
            y_local = self.net_local(X_local)
            y_high = self.net_high(X_high)
            y_global = self.net_global(X_global)
            X_final = torch.cat((y_local, y_high, y_global), dim=1)
            y = self.net_final(X_final)
            #
            loss = criterion(y, y0)
            optimizer.zero_grad()   # suggested trick
            loss.backward()
            optimizer.step()
            if epoch % 256 == 0:
                print 'epoch %s, loss %s'%(epoch, loss.data.cpu().numpy()[0])

        # test
        _X_local = np.array(self._X_local)[test_idx]
        _X_high = np.array(self._X_high)[test_idx]
        _X_global = np.array(self._X_global)[test_idx]
        _y0_flat = np.float32(self._y0).flatten()[test_idx]
        _y_flat = np.float32(self.predict(_X_local, _X_high, _X_global)).flatten()
        print self.__class__.__name__ + '.train: training finished. evaluation on last items: \n actual | predicted'
        for a, b in zip(_y0_flat, _y_flat):
            print a, b


    def parse_predict(self, gen, cell, makeparam):
        pass  # 未完待续

    def predict(self, _X_local, _X_high, _X_global):
        # pipeline
        _X_high = self.X_high_pipeline.transform(_X_high)
        # ann
        self.net_local.eval()
        self.net_high.eval()
        self.net_global.eval()
        self.net_final.eval()
        X_local= Variable(torch.FloatTensor(_X_local), requires_grad=True).cuda()
        X_high= Variable(torch.FloatTensor(_X_high), requires_grad=True).cuda()
        X_global= Variable(torch.FloatTensor(_X_global), requires_grad=True).cuda()
        #
        y_local = self.net_local(X_local)
        y_high = self.net_high(X_high)
        y_global = self.net_global(X_global)
        X_final = torch.cat((y_local, y_high, y_global), dim=1)
        y = self.net_final(X_final)
        # pipeline
        _y_inverse = self.y_pipeline.inverse_transform(y.data.cpu().numpy())
        return _y_inverse
