#!/usr/bin/python
import numpy as np
from scipy.optimize import minimize

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

from cStringIO import StringIO
from fuzzywuzzy import process


# qchem package
import qchem
import shared
import engine
from shared import ELEMENTS

qchem.Load()

def get_error(x):
    import numpy as np

    def pre_optimize_regular(cell, a, b, c, d, e, f):     # start ccoor, parameters
        import numpy as np
        ecell = engine.Cell(str(cell))
        sc = ecell.ccoor
        ec = np.copy(sc)
        for i, j in np.ndindex(len(sc), len(sc)):
            if i==j: continue
            x = sc[j] - sc[i]
            r = np.linalg.norm(x)
            ec[i] += x / r * a * np.exp(-b * r) * (c * r + d + e / r + f / r**2 )
        ecell.ccoor = ec
        return ecell

    cur_list = [
                # ['master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0','master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0'],
                # ['master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.10alt -0_02 opt','master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.10alt -0_02 opt'],
                # ['master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.20alt -0_02 opt','master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.20alt -0_02 opt contd'],
                # ['master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_02 opt','master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_02 opt'],
                # ['master.PbS QD.bare qd testing.Q0 Reproduce.Pb108S108.para opt','master.PbS QD.bare qd testing.Q0 Reproduce.Pb108S108.g opt'],
                # ['master.PbS QD.bare qd testing.Q0 Reproduce.Pb68S68.opt','master.PbS QD.bare qd testing.Q0 Reproduce.Pb68S68.g opt'],
                # ['master.PbS QD.bare qd testing.Q0 Reproduce.Pb63S62.opt','master.PbS QD.bare qd testing.Q0 Reproduce.Pb63S62.g opt'],
                ['master.PbS QD.bare qd testing.crunchit.3 opt','master.PbS QD.bare qd testing.crunchit.3 opt'],
                ['master.PbS QD.bare qd testing.crunchit.4 opt','master.PbS QD.bare qd testing.crunchit.4 opt'],
                ['master.PbS QD.bare qd testing.crunchit.5 opt','master.PbS QD.bare qd testing.crunchit.5 opt'],
                # ['master.PbS QD.bare qd testing.crunchit.6 opt','master.PbS QD.bare qd testing.crunchit.6 opt'],
                ['master.PbS QD.bare qd testing.crunchit.7 opt','master.PbS QD.bare qd testing.crunchit.7 opt']
               ]

    a,b,c,d,e,f = x
    error = 0

    for p in cur_list:

       scell = engine.Map().lookup(p[0]).cell
       ecell = engine.Map().lookup(p[1]).vasp.optimized_cell
       scell.recompute()
       ecell.recompute()
       error += engine.compare_cell_bijective(pre_optimize_regular(scell, a, b, c, d, e, f), ecell, suppress_output=True)

    with open(__file__+'.log', "a") as of:
        of.write('%s %s %s %s %s %s %s\n' %(a, b, c, d, e, f, error))
    return error

with open(__file__+'.log', "a") as of:
    of.write('%s\n' %(minimize(get_error, method='Powell', x0=[  6.155e-01,   1.783e+00,  -6.703e+01,   3.578e+01,   9.838e+02, -7.022e+02])))
#[0.03, 0.3, 0.1, 0.2, 0.3, 0.5]
#[  6.155e-01,   1.783e+00,  -6.703e+01,   3.578e+01,   9.838e+02, -7.022e+02]
