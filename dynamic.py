#!/usr/bin/env python
# -*- coding: utf-8 -*-
# scientific kit
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import matplotlib.pyplot as plt

# misc
import os
import time
import dill as pickle
import IPython
from tqdm import tqdm, trange
tqdm.monitor_interval = 0
import itertools
import math
import types
import copy
from collections import OrderedDict
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
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# parallel
from joblib import Parallel, delayed

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
    global nodes, services
    nodes = load('nodes')
    services = load('services')
    print 'global_load called'

def global_save():
    global nodes, services
    save(nodes, 'nodes')
    save(services, 'services')
    print 'global_save called'

# nodes
# ==============================================================================

nodes = {}

# ML
# ==============================================================================

services = {}

# MC_gen_queuetime
# ==============================================================================
