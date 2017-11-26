dynamic.global_load()
m = dynamic.MlPbSOptFCE()
parent = engine.Map().lookup('master.PbS QD.bare qd testing.ml_force')
for n in parent.map:
    if n.moonphase() == 2:
        m.parse_train(n.vasp)

m.train()

for i in range(len(m._X1)-50, len(m._X1)):
    print m._y0[i], m.predict_f(m._X1[i])
