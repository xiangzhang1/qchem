import numpy as np
from pprint import pprint

n = engine.Map().lookup('master.PbS QD.bare qd testing.crunchit')

for i in ['static', 'ncl']:
    a = np.zeros((5,5))
    b = np.zeros(5)
    for j in range(3,8):
        m = n.map.lookup(str(j) + ' ' + i)
        eline = [l for l in m.vasp.log.splitlines() if '1 F=' in l][0]
        e = float(eline.split()[4])
        a[j-3] = [j*j*j, 6*j*j, 12*j, 8, j%2]
        b[j-3] = e
        print j*j*j, 'e_bulk + ', 6*j*j, 'e_facet + ', 12*j, 'e_ridge + ', '8e_vertex + ', j%2, 'e_offstoi = ', e
    print np.linalg.solve(a,b)
