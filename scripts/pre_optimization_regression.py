#!/usr/bin/python
import numpy as np
from scipy.minimize import optimize


def pre_optimize_regular(cell, a, b, c, d, e, f):     # start ccoor, parameters

    sc = cell.ccoor
    ec = np.copy(sc)

    for i, j in np.ndindex(len(sc), len(sc)):

        if i==j: continue

        x = sc[j] - sc[i]
        r = norm(x)

        ec[i] += x / r * a * np.exp(-b * r) * (c * r + d + e / r + f / r**2 )

    cell.ccoor = ec







def get_error(a, b, c, d, e, f):

    cur_list = [
                ['master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0','master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0'],
                ['master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed | end - -0_02','master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed | end - -0_02'],
                ['master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed #2 | end - -0_02','master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed #2 | end - -0_02'],
                ['master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.10alt -0_02 opt','master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.10alt -0_02 opt'],
                ['master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.20alt -0_02 opt','master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.20alt -0_02 opt contd'],
                ['master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_02 opt','master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_02 opt'],
                ['master.PbS QD.bare qd testing.Q0 Reproduce.Pb108S108.para opt','master.PbS QD.bare qd testing.Q0 Reproduce.Pb108S108.g opt'],
                ['master.PbS QD.bare qd testing.Q0 Reproduce.Pb68S68.opt','master.PbS QD.bare qd testing.Q0 Reproduce.Pb68S68.g opt'],
                ['master.PbS QD.bare qd testing.Q0 Reproduce.Pb63S62.opt','master.PbS QD.bare qd testing.Q0 Reproduce.Pb63S62.g opt'],
                ['master.PbS QD.bare qd testing.crunchit.3 opt','master.PbS QD.bare qd testing.crunchit.3 opt'],
                ['master.PbS QD.bare qd testing.crunchit.4 opt','master.PbS QD.bare qd testing.crunchit.4 opt'],
                ['master.PbS QD.bare qd testing.crunchit.5 opt','master.PbS QD.bare qd testing.crunchit.5 opt'],
                ['master.PbS QD.bare qd testing.crunchit.6 opt','master.PbS QD.bare qd testing.crunchit.6 opt'],
                ['master.PbS QD.bare qd testing.crunchit.7 opt','master.PbS QD.bare qd testing.crunchit.7 opt'],
               ]

       error = 0

       for p in cur_list:

           scell = engine.Map().lookup(p[0]).cell
           ecell = engine.Map().lookup(p[1]).vasp.optimized_cell

           pre_optimize_regular(scell, a, b, c, d, e, f)

           error += engine.compare_cell_bijective(scell, ecell)

       return error





print minimize(get_error, x0=[0.07, 0.3, 0.1, 0.2, 0.3, 0.5])
