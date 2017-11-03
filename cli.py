#!/usr/bin/python
'''
A wrapper for IPython as well as a boilerplate.
'''

# common libraries
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import shutil
import random
import string
import dill as pickle
import time
from pprint import pprint
import IPython
import numpy as np

from cStringIO import StringIO
from fuzzywuzzy import process

import tensorflow as tf


# qchem package
import qchem
import shared
import dynamic
import engine
from shared import ELEMENTS

print shared.bcolors.OKBLUE + 'Welcome. Libraries loaded.' + shared.bcolors.ENDC
# IPython.embed(banner1='')

dynamic.global_load()
self = dynamic.MlVaspMemory()
self.fit_B()
self.init_data()
self.fit()
