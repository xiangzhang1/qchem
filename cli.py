#!/usr/bin/python
'''
An environment. Do not paste actions here.
'''

# common libraries
import sys
import os
import shutil
import random
import string
import dill as pickle
import time
from pprint import pprint
import IPython
import numpy as np
np.set_printoptions(precision=4, threshold='nan')
from tqdm import tqdm
import math
import itertools

from cStringIO import StringIO
from fuzzywuzzy import process
import types

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# scipy
from scipy import spatial
from scipy.optimize import minimize

# sklearn
from sklearn.preprocessing import StandardScaler

# pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.nn import Sequential

# qchem package
import qchem
import shared
import dynamic
import engine
from shared import ELEMENTS

print shared.bcolors.OKBLUE + 'Welcome. Libraries loaded.' + shared.bcolors.ENDC

IPython.embed(banner1='')
