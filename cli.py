#!/usr/bin/env python
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

# ase: some of its libraries are fantastic.
import ase

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

# qchem package
import qchem
import shared
import dynamic
import engine
from shared import elements

print shared.bcolors.OKBLUE + 'Welcome. Libraries loaded.' + shared.bcolors.ENDC

if __name__ == "__main__":
    IPython.embed(banner1='')
