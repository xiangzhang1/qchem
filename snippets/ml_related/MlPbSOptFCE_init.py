dynamic.global_load()
m = dynamic.MlPbSOptFCE()
parent = engine.Map().lookup('master.PbS QD.bare qd testing.ml_force')
for n in parent.map:
    if n.moonphase() == 2:
        m.parse_train(n.vasp)

m.train()
