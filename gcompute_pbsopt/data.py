dynamic.global_load()

# any more-than-125 same-length from-scratch training produces 0.
# incremental training is no better.
# lowering learning rate seems to cause a significant improvement. presumably the idea is: the error goes to 225 because the calculation 'blows up' irreversibly. however, as soon as error goes to ~144, explosion happens.
# trying changing to linera activation does not end well. nor does extensively changing model structure.
#   tanh activation (all-LSTM or 1-LSTM): results in step function. presumably because summation is allowed.
#   elu and relu (all-LSTM) blows up.

curs = [
['master.PbS QD.bare qd testing.crunchit.3 opt','master.PbS QD.bare qd testing.crunchit.3 opt'],
['master.PbS QD.bare qd testing.crunchit.4 opt','master.PbS QD.bare qd testing.crunchit.4 opt'],
['master.PbS QD.bare qd testing.crunchit.5 opt','master.PbS QD.bare qd testing.crunchit.5 opt'],
['master.PbS QD.bare qd testing.crunchit.6 opt','master.PbS QD.bare qd testing.crunchit.6 opt'],
['master.PbS QD.bare qd testing.crunchit.7 opt','master.PbS QD.bare qd testing.crunchit.7 opt'],
['master.PbS QD.bare qd testing.crunchit.8 opt half-complete','master.PbS QD.bare qd testing.crunchit.8 opt0'],
['master.PbS QD.bare qd testing.ml_opt.Pb48S48 rect','master.PbS QD.bare qd testing.ml_opt.Pb48S48 rect'],
['master.PbS QD.bare qd testing.ml_opt.Pb72S72 rect','master.PbS QD.bare qd testing.ml_opt.Pb72S72 rect'],
['master.PbS QD.bare qd testing.ml_opt.Pb36S36 rect','master.PbS QD.bare qd testing.ml_opt.Pb36S36 rect'],
['master.PbS QD.bare qd testing.ml_opt.Pb40S40 customshape','master.PbS QD.bare qd testing.ml_opt.Pb40S40 customshape'],
['master.PbS QD.bare qd testing.ml_opt.Pb80S80 1 13','master.PbS QD.bare qd testing.ml_opt.Pb80S80 1 13'],
['master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0','master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0']
]

m = dynamic.MlPbSOptXRNN()
for bcur, ecur in curs:
    bn = engine.Map().lookup(bcur)
    en = engine.Map().lookup(ecur)
    m.parse_train(bn, en)

m.train(batch_size=64, epochs=1)
