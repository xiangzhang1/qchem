import numpy as np
import os

qchem.Load()

p = engine.Map().lookup('master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38')
p.map.add_edge('10alt -0_02 opt', 'compare cell')
n = p.map.lookup('compare cell')

n.reset()
n.property = 'engine=md, movie'
n.compute(angle=1)
