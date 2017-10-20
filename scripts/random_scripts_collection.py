import numpy as np
import os

p = engine.Map().lookup('master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38')
p.map.add_edge('20alt -0_02 opt contd', 'compare cell')
n = p.map.lookup('compare cell')

n.property = 'engine=md, movie, elev=10, angle=1'
n.reset()
n.compute()







# when upgrading versions, data structure can change. use the __str__ feature to upgrade the data structure smoothly.
def upgrade_recompute(node):
    '''
    This version is for recomputing cell.
    Note that both node.cell and vasp.optimized_cell must be recomputed.
    '''
    try:
        node.cell = engine.Cell(str(node.cell))
    except AttributeError:
        pass
    try:
        node.vasp.optimized_cell = engine.Cell(str(node.vasp.optimized_cell))
    except AttributeError:
        pass

l = engine.Map().lookup('master').map.traverse()
for n in l:
    upgrade_recompute(n)
