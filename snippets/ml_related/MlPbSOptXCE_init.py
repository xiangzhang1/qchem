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

m = dynamic.MlPbSOptXRNN()
for cur in curs:
    n = engine.Map().lookup(cur)
    m.parse_train(n.vasp)
    
m.train()

# learning_rate
#   1-0.1: to average
#   0.1 - 1E-3: roughly same
#   1E-4: no learning

# convergence:
#   total_set_size=150, 1500 epochs:
#       significantly worse than linear regression [possibly due to indirect summation]
#       hard cap w.r.t. time met



train_err = np.linalg.norm([m._y0[i] - m.predict(m._X1[i]) for i in range(0, len(m._X1)-50)])
test_err = np.linalg.norm([[list(m._y0[i]), list(m.predict(m._X1[i]))] for i in range(len(m._X1)-50, len(m._X1))])
print n_epochs, train_err, test_err
