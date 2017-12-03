#!/usr/bin/env python
import dynamic
import engine
import qchem

import tensorflow as tf
import IPython
import numpy as np
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer, StandardScaler
import dill as pickle


np.set_printoptions(precision=2, threshold='nan', suppress=True, linewidth=1200)

dynamic.global_load()

curs = [
    ['master.PbS QD.bare qd testing.crunchit.3 opt','master.PbS QD.bare qd testing.crunchit.3 opt'],
    ['master.PbS QD.bare qd testing.crunchit.4 opt','master.PbS QD.bare qd testing.crunchit.4 opt'],
    ['master.PbS QD.bare qd testing.crunchit.5 opt','master.PbS QD.bare qd testing.crunchit.5 opt'],
    ['master.PbS QD.bare qd testing.crunchit.6 opt','master.PbS QD.bare qd testing.crunchit.6 opt'],
    ['master.PbS QD.bare qd testing.crunchit.7 opt','master.PbS QD.bare qd testing.crunchit.7 opt'],
    ['master.PbS QD.bare qd testing.crunchit.8 opt half-complete','master.PbS QD.bare qd testing.crunchit.8 opt0'],
    ['master.PbS QD.bare qd testing.ml_opt.Pb48S48 rect','master.PbS QD.bare qd testing.ml_opt.Pb48S48 rect'],
    ['master.PbS QD.bare qd testing.ml_opt.Pb72S72 rect','master.PbS QD.bare qd testing.ml_opt.Pb72S72 rect'],
    ['master.PbS QD.bare qd testing.ml_opt.Pb36S36 rect','master.PbS QD.bare qd testing.ml_opt.Pb36S36 rect'],
    ['master.PbS QD.bare qd testing.ml_opt.Pb40S40 customshape','master.PbS QD.bare qd testing.ml_opt.Pb40S40 customshape'],
    ['master.PbS QD.bare qd testing.ml_opt.Pb80S80 1 13','master.PbS QD.bare qd testing.ml_opt.Pb80S80 1 13'],
    ['master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0','master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0']
]


for bcur, ecur in curs:

    bn = engine.Map().lookup(bcur)
    bcell = bn.cell
    bcoor = bn.cell.ccoor
    en = engine.Map().lookup(ecur)
    ecoor = en.vasp.optimized_cell.ccoor
    natom0 = en.cell.stoichiometry.values()[0]
    natom = en.cell.natoms()
    a = 6.014 / 2

    _A = np.eye(natom)

    with open('test_neighbor_reversibility.txt', 'a') as f:
        for i, ci in enumerate(ecoor):
            for j, cj in enumerate(ecoor):
                if j != i and np.linalg.norm(cj-ci)<4.5:
                    _A[i, j] = -1
        f.write('+' * 200 + '\n')
        f.write("Eigenvals (note 0.j): %s\n" %np.linalg.eigvals(_A))
