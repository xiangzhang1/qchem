# any more-than-125 same-length from-scratch training produces 0.
# incremental training is no better.
# lowering learning rate seems to cause a significant improvement. presumably the idea is: the error goes to 225 because the calculation 'blows up' irreversibly. however, as soon as error goes to ~144, explosion happens.
# trying changing to linear activation does not end well. nor does extensively changing model structure.
#   tanh activation (all-LSTM or 1-LSTM): results in step function. presumably because summation is allowed.
#   elu and relu (all-LSTM) blows up.
import gc
import os

dynamic.global_load()

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

eval_curs = [
    ['master.PbS QD.bare qd testing.crunchit.6 opt','master.PbS QD.bare qd testing.crunchit.6 opt'],
    ['master.PbS QD.bare qd testing.ml_opt.Pb40S40 customshape','master.PbS QD.bare qd testing.ml_opt.Pb40S40 customshape'],
    ['master.PbS QD.bare qd testing.ml_opt.Pb80S80 1 13','master.PbS QD.bare qd testing.ml_opt.Pb80S80 1 13']
]

for eval_pair in eval_curs:
    train_curs = [pair for pair in curs if pair != eval_pair]

    fig, ax = plt.subplots()
    filename = 'MlPbSOptXLasso-test_%s.png' %(eval_pair[0])

    m = dynamic.MlPbSOptXLasso()
    for bcur, ecur in train_curs:
        print 'parsing node %s' %bcur
        bn = engine.Map().lookup(bcur)
        en = engine.Map().lookup(ecur)
        m.parse_train(bn, en)
    print 'training'
    m.train()
    print 'training complete'
    y0, y = m.eval()
    ax.scatter(y0, y, s=0.5, label='train set')

    # eval
    m._X = [] ; m._y0 = []
    bcur, ecur = eval_pair
    bn = engine.Map().lookup(bcur)
    en = engine.Map().lookup(ecur)
    m.parse_train(bn, en)
    y0, y = m.eval()
    ax.scatter(y0, y, s=0.5, label='test set %s' %bcur)

    fig.legend()
    plt.title('train-test-splitted testing of %s' %bcur)
    plt.savefig(filename, format='png')
    print 'time %s, saving %s' %(str(datetime.now()), filename)
