import matplotlib.pyplot as plt
import numpy as np

n1 = engine.Map().lookup('master.PbS QD.bare qd testing.crunchit.7 opt')
n2 = engine.Map().lookup('master.PbS QD.bare qd testing.crunchit.7 opt')
c1 = n1.cell.ccoor
c2 = n2.vasp.optimized_cell.ccoor
dc = c2-c1

center = np.mean(c2, axis=0)
indices = [idx for idx,c in enumerate(c2) if (c - center + [0.4,0.4,0.4] > 0).all()]

fig = plt.figure()
ax = fig.gca(projection='3d')

plt.title(n1.name)
q = ax.quiver(c1[indices,0], c1[indices,1], c1[indices,2], dc[indices,0], dc[indices,1], dc[indices,2])

plt.show()

---

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.quiver(bcoor[:,0], bcoor[:,1], bcoor[:,2], dc[:,0], dc[:,1], dc[:,2])
plt.show()
