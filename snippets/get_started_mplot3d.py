import matplotlib.pyplot as plt
import numpy as np

n1 = engine.Map().lookup('master.PbS QD.bare qd testing.crunchit.5opt')
n2 = engine.Map().lookup('master.PbS QD.bare qd testing.crunchit.5opt reversePbS')
c1 = n1.cell.ccoor
c2 = n2.vasp.optimized_cell.ccoor
dc = c2-c1

fig = plt.figure()
ax = fig.gca(projection='3d')

plt.title(n1.name)
q = ax.quiver(c1[:,0], c1[:,1], c1[:,2], dc[:,0], dc[:,1], dc[:,2])

plt.show()
