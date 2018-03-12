#!/usr/bin/env python
'''Deprecated. I used to use this one for GUI, but after discovering Jupyter, IPython just doesn't quite cut it.'''
# scientific kit
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
from scipy import spatial
from scipy.optimize import minimize
import pandas as pd
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt

# date
import time
import datetime

# sys
import os
import sys
import subprocess
import shutil

import sys
import os
import shutil
import random
import string
import dill as pickle
import time
from pprint import pprint
import IPython
from tqdm import tqdm
import math
import itertools
import copy
import datetime

from cStringIO import StringIO
from fuzzywuzzy import process
import types

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# sklearn
import sklearn
from sklearn.preprocessing import StandardScaler

# keras
import keras

# parallel
import joblib
from joblib import Parallel, delayed

# import ase

# qchem package
from qchem import shared, dynamic, engine, graph



