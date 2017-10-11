import numpy as np
import os

qchem.Load()

p = engine.Map().lookup('master.PbS QD.bare qd testing.crunchit')
p.map.add_edge('md', '5 opt')
n = p.map.lookup('md')

n.property = 'engine=md, movie, elev=10, angle=1'
n.reset()
n.compute()
