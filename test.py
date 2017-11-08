#!/usr/bin/python
'''
An environment. Do not paste actions here.
'''

# common libraries
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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


sys.stdout = open('test.py.log', 'w')








# -------------------------------------------------------------

curs = [
# # Reproduce
#     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb108S108.g opt',
#     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb68S68.g opt',
#     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb63S62.g opt',
#     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb55S38.g opt',
# # Test convergence
#     'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0',
#     # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02',
#     # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed | end - -0_02',
#     # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed #2 | end - -0_02',
# # Crunchit
#     'master.PbS QD.bare qd testing.crunchit.3 opt',
#     'master.PbS QD.bare qd testing.crunchit.4 opt',
#     'master.PbS QD.bare qd testing.crunchit.5 opt',
#     'master.PbS QD.bare qd testing.crunchit.6 opt',
    'master.PbS QD.bare qd testing.crunchit.7 opt',
]

dynamic.global_load()
m = dynamic.MlPbSOpt()
for cur in curs:
    n = engine.Map().lookup(cur)
    print 'parsing cur %s' %cur
    m.parse_obj(n.vasp)
    print 'parsing complete.'
dynamic.save(m, m.__class__.__name__)
m.train()
