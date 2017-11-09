SpeedNet#!/usr/bin/python
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
from tqdm import tqdm

from cStringIO import StringIO
from fuzzywuzzy import process

# pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# qchem package
import qchem
import shared
import dynamic
import engine
from shared import ELEMENTS


sys.stdout = open('optimize_MlVaspSpeed_parameters.log', 'w')

print shared.bcolors.OKBLUE + 'Welcome. Libraries loaded.' + shared.bcolors.ENDC

dynamic.global_load()
m = dynamic.MlVaspSpeed()
for n in engine.Map().lookup('master').map.traverse():
    try:
        n.cell = engine.Cell(str(n.cell))
        n.gen.cell = n.cell
        n.vasp.cell = n.cell
        n.vasp.gen = n.gen
        n.vasp.optimized_cell = engine.Cell(str(n.vasp.optimized_cell))
    except AttributeError:
        pass
    if getattr(n, 'gen', None) and n.gen.parse_if('engine=vasp') and n.moonphase()==2:
        try:
            m.parse_obj(n.vasp, engine.Makeparam(n.vasp.gen))
        except (shared.CustomError, shared.DeferError) as e:
            print 'warning: node %s\'s parsing failed. probably old version.' %n.name ; sys.stdout.flush()
m.train()



# The following is a little deprecated.

def f(x, m=m, optimizer_name='SGD'):    #  train then return error. for hyperparameter search.
    print '----------------------------' ; sys.stdout.flush()
    x = abs(x)
    bn_momentum, dropout_p, learning_rate, batch_size, n_epochs = x[0] / 10.0, x[1] / 15.0, 10**(-1*x[2]), int(10 * x[3]), int(1000 * x[4])
    m.net = dynamic.MlVaspSpeedNet(bn_momentum=bn_momentum, dropout_p=dropout_p)
    err = m.train(learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs, optimizer_name=optimizer_name)
    print 'parameters: %s. error: %s.' %(x, err) ; sys.stdout.flush()
    return err

# f([9, 1, 2, 3.2, 4])

from scipy.optimize import minimize
print minimize(f, x0=np.float32([9, 1, 2, 3.2, 4]), method='Powell') ; sys.stdout.flush()
# 9.056  0.017  1.726  3.2    4. -> better retain some dropout   |    0.9, 0.06, 0.02, 32, 4000
print 'finished! :)'
