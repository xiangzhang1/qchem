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

    def __init__(self, bn_momentum=0.74, dropout_p=0.07):
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


    def parse_train(self, vasp, gen, cell, makeparam):
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
        self._X.append([
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
        ])
        self._y0.append([time_elec_step])   # put it here so that no inconsistency will happen

    def train(self, n_epochs=4000, batch_size=32, learning_rate=0.01, optimizer_name='SGD', test_set_size=5):
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


# inital training script for MlVaspSpeed ca nbe found in scripts/machine_learning_benchmark/optimize_MlVaspSpeed_parameters.py






# MlPbSOpt
# ==============================================================================


class MlPbSOptNet(nn.Module):

    def __init__(self, bn_momentum=0.74, dropout_p=0.07):   # 2500 variables in total. Expect high dropout.
        super(MlPbSOptNet, self).__init__()
        self.lA1 = nn.Linear(125, 15)
        self.bnA1 = nn.BatchNorm1d(15, momentum=bn_momentum)
        self.dropoutA1 = nn.Dropout(p=dropout_p)
        self.lA2 = nn.Linear(15, 10)
        self.bnA2 = nn.BatchNorm1d(10, momentum=bn_momentum)
        self.dropoutA2 = nn.Dropout(p=dropout_p)

        self.lB1 = nn.Linear(6, 8)
        self.bnB1 = nn.BatchNorm1d(8, momentum=bn_momentum)
        self.dropoutB1 = nn.Dropout(p=dropout_p)
        self.lB2 = nn.Linear(8, 5)
        self.bnB2 = nn.BatchNorm1d(5, momentum=bn_momentum)
        self.dropoutB2 = nn.Dropout(p=dropout_p)

        self.lC1 = nn.Linear(20, 10)
        self.bnC1 = nn.BatchNorm1d(10, momentum=bn_momentum)
        self.dropoutC1 = nn.Dropout(p=dropout_p)
        self.lC2 = nn.Linear(10, 5)
        self.bnC2 = nn.BatchNorm1d(5, momentum=bn_momentum)
        self.dropoutC2 = nn.Dropout(p=dropout_p)

        self.l1 = nn.Linear(20, 5)
        self.bn1 = nn.BatchNorm1d(5, momentum=bn_momentum)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.l2 = nn.Linear(5, 5)
        self.bn2 = nn.BatchNorm1d(5, momentum=bn_momentum)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.l3 = nn.Linear(5, 1)

    def forward(self, X):   # 啊！真舒畅！

        A = self.bnA1(self.dropoutA1(F.elu(self.lA1(X[:, :125]))))
        A = self.bnA2(self.dropoutA2(F.elu(self.lA2(A))))

        B = self.bnB1(self.dropoutB1(F.elu(self.lB1(X[:, 125:125+6]))))
        B = self.bnB2(self.dropoutB2(F.elu(self.lB2(B))))

        C = self.bnC1(self.dropoutC1(F.elu(self.lC1(X[:, 125+6:125+6+20]))))
        C = self.bnC2(self.dropoutC2(F.elu(self.lC2(C))))

        y = torch.cat((A, B, C), dim=1)
        y = self.bn1(self.dropout1(F.elu(self.l1(y))))
        y = self.bn2(self.dropout2(F.elu(self.l2(y))))
        y = self.l3(y)

        return y


class MlPbSOpt(object):

    def __init__(self):
        # data
        self._X = []
        self._y0 = []
        # pipeline
        self.X_pipeline = StandardScaler()
        self.y_pipeline = StandardScaler()
        # ann. what a pity.
        self.net = MlPbSOptNet()


    def parse_train(self, vasp):
        a = 6.01417/2
        cell = vasp.optimized_cell

        # first, we get the fcoor and use that. no rotation need be taken into account.
        ccoor = cell.ccoor
        def error_after_transformation(transformation, ccoor=ccoor, a=a):       # note: parallelization doesn't save time.
            x0, y0, z0, sz, sx, sy = transformation
            rotation_matrix = shared.euler2mat(sz, sx, sy)
            ccoor_transformed = np.dot(np.subtract(ccoor, [x0, y0, z0]), rotation_matrix.T)
            fcoor = ccoor_transformed / a
            return np.linalg.norm(fcoor - np.around(fcoor))
        t = minimize(fun=error_after_transformation,
                     x0=[0, 0, 0, 0, 0, 0],
                     bounds=[(-.8,.8), (-.8,.8), (-.8,.8), (-.2,.2), (-.2,.2), (-.2,.2)]
                    ).x
        x0, y0, z0, sz, sx, sy = transformation
        rotation_matrix = shared.euler2mat(sz, sx, sy)
        ccoor_transformed = np.dot(np.subtract(ccoor, [x0, y0, z0]), rotation_matrix.T)
        fcoor = ccoor_transformed / a

        # second, we convert the sparse matrix to a dense matrix
        Nx = math.ceil(np.linalg.norm(cell.base[0]) / a)      # assume 长方体
        Ny = math.ceil(np.linalg.norm(cell.base[1]) / a)
        Nz = math.ceil(np.linalg.norm(cell.base[2]) / a)
        dense_matrix = np.zeros((Nx, Ny, Nz, 4))
        for idx_atom, fc in fcoor:
            # dense matrix 指标
            ix, iy, iz = np.around(fc)
            # 符号位
            idx_ele = 0 if idx_atom < cell.stoichiometry.values()[0] else 1     # OrderedDict, 品质保证!     # assume Pb S only
            symbol = cell.stoichiometry.keys()[idx_ele]
            feature_n = 2 if symbol=='Pb' else -2
            dense_matrix[ix, iy, iz, 0] = feature_n
            # 数值位: dx, dy, dz
            dense_matrix[ix, iy, iz, 1:] = fc - np.around(fc)

        # fifth, one vasp produces many input. we should utilize all of them. we do this by generating every possible permutation of dense_matrix
        dense_matrices  = [dense_matrix[::reverse_x, ::reverse_y, ::reverse_z].transpose(order) for reverse_x in [-1,1] for reverse_y in [-1,1] for reverse_z in [-1,1] for order in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,1,0),(2,0,1)]]
        for dense_matrix in dense_matrices:

            # third, we parse features and labels from the dense matrix.
            # --pre-parsing the convex-hull--
            hull = ConvexHull(np.around(fcoor))
            nvertices = len(hull.vertices)
            if nvertices > 20:  raise shared.CustomError(self.__class__.__name__ + '.parse_train: # vertices > 20. Assumption broken. Rethink.')
            vertice_coordinates = np.float32([fcoor[iv] for iv in hull.vertices])
            center_coordinate = np.mean(fcoor, axis=0)
            # -------------------------------
            for idx_atom, fc in fcoor:
                # dense matrix 空降！
                ix, iy, iz = np.around(fc)
                # 拔剑四顾！
                feature_npart = dense_matrix[ix-2:ix+3, iy-2:iy+3, iz-2:iz+3, 0].flatten()    # 您点的5*5*5矩阵到货啦！      # C式拍平，质量保证！
                # 还有点小尾巴: stoichiometry 以及 convexhull
                feature_stoichiometry = np.float32([cell.stoichiometry['Pb'], cell.stoichiometry['S']])
                displace_to_center = fc - center_coordinate
                dist_to_vertices = np.sum((vertice_coordinates - fc)**2,axis=1)**(1./2)
                np.sort(dist_to_vertices) ; dist_to_vertices = np.pad(dist_to_vertices, (0, 20-nvertices))

                # fourth, formally establish features and labels
                _X = np.concatenate((feature_npart, feature_stoichiometry, displace_to_center, dense_matrix[ix:ix+1,iy:iy+1,iz:iz+1,0:1], dist_to_vertices))    # 125 + (2 + 3 + 1) + (20)
                _y0 = (fc - np.around(fc))[0]



    def train(self, n_epochs=1024, batch_size=128, learning_rate=0.01, optimizer_name='SGD', test_set_size=128):
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
            if epoch % 128 == 0:
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
        pass  # 未完待续

    def predict(self, _X):
        # pipeline
        _X = self.X_pipeline.transform(_X)
        # ann
        self.net.eval()
        y = self.net(Variable(torch.FloatTensor(_X), requires_grad=True))
        # pipeline
        _y_inverse = self.y_pipeline.inverse_transform(y.data.numpy())
        return _y_inverse
