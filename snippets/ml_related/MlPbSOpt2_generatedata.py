n = engine.Map().lookup('master.PbS QD.bare qd testing.ml_force.Pb72S72rect wiggle0')
parent = engine.Map().lookup('master.PbS QD.bare qd testing.ml_force')
for i in range(8):
    n_copy = n.copy()
    n_copy.cell.ccoor += np.random.uniform(low=-0.35, high=0.35, size=n_copy.cell.ccoor.shape)
    parent.map.add_node(n_copy)
    n_copy.compute()
