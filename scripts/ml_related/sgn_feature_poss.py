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

def udf_nn(*args):
    layers = []
    for i in range(len(args)-1):
        layers.append(nn.Linear(args[i], args[i+1]))
        if i != len(args)-2:
            layers.append(nn.ELU())
    return Sequential(*layers)




def f(sgn1, sgn2, x, y, z):
    multiplier = 3 if sgn1==1 and sgn2==1 else -2
    return multiplier / math.sqrt(x**2 + y**2 + z**2)

_X = []
_y0 = []
for i in range(0,500):
    sgn1, sgn2 = np.random.choice([-1,1]), np.random.choice([-1,1])
    x, y, z = np.random.rand(3)
    _X.append([sgn1, sgn2, x, y, z])
    _y0.append([f(sgn1, sgn2, x, y, z)])
_X = np.array(_X)
_y0 = np.array(_y0)

net = udf_nn(5, 80, 10, 1)
# net = udf_nn(5, 5, 4, 4, 3, 3, 2, 1)

n_epochs=900
batch_size=64
learning_rate=0.026
test_set_size=50
test_idx = np.random.choice(range(len(_X)), size=test_set_size)
train_idx = np.array([i for i in range(len(_X)) if i not in test_idx])
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# train
net.train()
for epoch in range(n_epochs):
    batch_idx = np.random.choice(range(_X.shape[0]), size=batch_size)
    X_batch= Variable(torch.FloatTensor(_X[batch_idx]), requires_grad=True)
    y0_batch = Variable(torch.FloatTensor(_y0[batch_idx]), requires_grad=False)
    y = net(X_batch) * X_batch[:, 0:1] * X_batch[:, 1:2]
    loss = criterion(y, y0_batch)
    optimizer.zero_grad()   # suggested trick
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print 'epoch %s, loss %s'%(epoch, loss.data.numpy()[0])

# test
net.eval()
_X = Variable(torch.FloatTensor(_X[test_idx]), requires_grad=True)
_y0 = np.array(_y0)[test_idx].flatten()
_y = (net(_X) * _X[:, 0:1] * _X[:, 1:2]).data.numpy().flatten()
print '.train: training finished. evaluation on last items: \n actual | predicted'
for a, b in zip(_y0, _y):
    print a, b
