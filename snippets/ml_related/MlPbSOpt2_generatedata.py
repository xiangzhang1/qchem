n = engine.Map().lookup('master.PbS QD.bare qd testing.ml_force.7static wiggle0')
parent = engine.Map().lookup('master.PbS QD.bare qd testing.ml_force')
for i in range(10):
    n_copy = n.copy()
    n_copy.cell.ccoor += np.random.uniform(low=-0.25, high=0.25, size=n_copy.cell.ccoor.shape)
    parent.map.add_node(n_copy)
    n_copy.compute()
