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


sys.stdout = open('optimize_MlPbSOpt_parameters.log', 'w')

    curs = [
    # # Reproduce
    #     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb108S108.g opt',
    #     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb68S68.g opt',
    #     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb63S62.g opt',
    #     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb55S38.g opt',
    # # # Test convergence
    #     'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0',
    # #     # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02',
    # #     # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed | end - -0_02',
    # #     # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed #2 | end - -0_02',
    # # Crunchit
        'master.PbS QD.bare qd testing.crunchit.3 opt',
        'master.PbS QD.bare qd testing.crunchit.4 opt',
        'master.PbS QD.bare qd testing.crunchit.5 opt',
        'master.PbS QD.bare qd testing.crunchit.6 opt',
        'master.PbS QD.bare qd testing.crunchit.7 opt',
    ]

    dynamic.global_load()
    m = dynamic.MlPbSOpt()
    for cur in curs:
        n = engine.Map().lookup(cur)
        print 'parsing cur %s' %cur
        m.parse_train(n.vasp)
        print 'parsing complete.'

n = engine.Map().lookup('master.PbS QD.bare qd testing.crunchit.7 opt')
vasp = n.vasp
cell = vasp.optimized_cell
# continue to dynmaic


def f(x, m=m, optimizer_name='SGD'):    #  train then return error. for hyperparameter search.
    print '----------------------------' ; sys.stdout.flush()
    x = abs(x)
    bn_momentum, dropout_p, learning_rate, batch_size, n_epochs = x[0] / 10.0, x[1] / 15.0, 10**(-1*x[2]), int(100 * x[3]), int(1000 * x[4])
    m.net = dynamic.MlPbSOptNet(bn_momentum=bn_momentum, dropout_p=dropout_p)
    err = m.train(learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs, optimizer_name=optimizer_name)
    print 'parameters: %s. error: %s.' %(x, err) ; sys.stdout.flush()
    return err

from scipy.optimize import minimize
print minimize(f, x0=np.float32([9, 1, 2, 1.2, 4]), method='Powell') ; sys.stdout.flush()
print 'finished! :)'
