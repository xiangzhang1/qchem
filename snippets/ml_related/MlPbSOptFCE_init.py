dynamic.global_load()

curs = [
'master.PbS QD.bare qd testing.crunchit.3 opt',
'master.PbS QD.bare qd testing.crunchit.4 opt',
'master.PbS QD.bare qd testing.crunchit.5 opt',
'master.PbS QD.bare qd testing.crunchit.6 opt',
'master.PbS QD.bare qd testing.crunchit.7 opt',
'master.PbS QD.bare qd testing.ml_opt.Pb48S48 rect',
'master.PbS QD.bare qd testing.ml_opt.Pb72S72 rect',
'master.PbS QD.bare qd testing.ml_opt.Pb36S36 rect'
]

m = dynamic.MlPbSOptFCE()
for cur in curs:
    n = engine.Map().lookup(cur)
    m.parse_train(n.vasp)

m.train()

for i in range(len(m._X1)-100, len(m._X1)-50):
    print m._y0[i], m.predict(m._X1[i])
