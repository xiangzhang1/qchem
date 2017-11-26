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



optimizer_name = 'Adam'
n_epochs = 1000

print '-' * 30 + ' %s ' %optimizer_name + '-' * 30

m = dynamic.MlPbSOptXCE()
for cur in curs:
    n = engine.Map().lookup(cur)
    m.parse_train(n.vasp)
m.train(n_epochs=n_epochs, optimizer_name=optimizer_name)
train_err = np.linalg.norm([m._y0[i] - m.predict(m._X1[i]) for i in range(0, len(m._X1)-50)])
test_err = np.linalg.norm([[list(m._y0[i]), list(m.predict(m._X1[i]))] for i in range(len(m._X1)-50, len(m._X1))])
print n_epochs, train_err, test_err
