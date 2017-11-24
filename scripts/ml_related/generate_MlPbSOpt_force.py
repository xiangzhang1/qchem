n = engine.Map().lookup('master.PbS QD.bare qd testing.ml_force.5static wiggle')
parent = engine.Map().lookup('master.PbS QD.bare qd testing.ml_force')
for i in range(5):
    n_copy = n.copy()
    n_copy.cell.ccoor += np.random.uniform(low=-0.35, high=0.35, size=n_copy.cell.ccoor.shape)
    parent.map.add_node(n_copy)
    n_copy.compute()
