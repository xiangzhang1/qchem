# any more-than-125 same-length from-scratch training produces 0.
# incremental training is no better.
# lowering learning rate seems to cause a significant improvement. presumably the idea is: the error goes to 225 because the calculation 'blows up' irreversibly. however, as soon as error goes to ~144, explosion happens.
# trying changing to linear activation does not end well. nor does extensively changing model structure.
#   tanh activation (all-LSTM or 1-LSTM): results in step function. presumably because summation is allowed.
#   elu and relu (all-LSTM) blows up.

from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking, Dropout, BatchNormalization, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
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

for batch_size in [256, 128, 32]:

    for eval_pair in eval_curs:
        train_curs = [pair for pair in curs if pair != eval_pair]

        for neighbor in [False, True]:

            for model_name, model in [
                ('original', Sequential([
                    LSTM(32, input_shape=(None, 4)),
                    Dense(24),
                    Dense(8),
                    Dense(1)
                ]))
                ('L32_N1', Sequential([
                    LSTM(32, input_shape=(None, 4)),
                    Dense(1)
                ])),
                ('L32_N4_N1', Sequential([
                    LSTM(32, input_shape=(None, 4)),
                    Dense(4, activation='elu'),
                    Dense(1, activation=None)
                ])),
                ('L32_L4_L1', Sequential([
                    LSTM(32, input_shape=(None, 4), return_sequences=True),
                    LSTM(4, return_sequences=True),
                    LSTM(1, activation=None)
                ])),
                ('L32_N4_N1_wlBD', Sequential([
                    LSTM(32, input_shape=(None, 4)),
                    Dropout(0.05),
                    Dense(4),
                    BatchNormalization(),
                    Activation('elu'),
                    Dropout(0.05),
                    Dense(1)
                ])),
                ('L32_N4_N4_N1_tanh', Sequential([
                    LSTM(32, input_shape=(None, 4)),
                    Dense(4, activation='tanh'),
                    Dense(4, activation='tanh'),
                    Dense(1)
                ])),  # simplest without linear?
            ]:

                fig, ax = plt.subplots()
                filename = 'MlPbSOptXRNN-batch_%s-neighbor_%s-model%s-test_%s.png' %(batch_size, neighbor, model_name, eval_pair[0])
                if os.path.exists(filename):
                    del model
                    gc.collect()
                    print 'skipping %s' %filename
                    continue
                os.mkdir(filename + '.dir')
                os.chdir(filename + '.dir')

                m = dynamic.MlPbSOptXRNN(model=model)
                for bcur, ecur in train_curs:
                    bn = engine.Map().lookup(bcur)
                    en = engine.Map().lookup(ecur)
                    m.parse_train(bn, en, neighbor=neighbor)
                m.train(batch_size=256, epochs=2048 / 256 * batch_size)     # on cloud
                # y0, y = m.eval()
                # ax.scatter(y0, y, s=0.5, label='train set')
                #
                # # eval
                # m._X = [] ; m._y0 = []
                # bcur, ecur = eval_pair
                # bn = engine.Map().lookup(bcur)
                # en = engine.Map().lookup(ecur)
                # m.parse_train(bn, en, neighbor=neighbor)
                # y0, y = m.eval()
                # ax.scatter(y0, y, s=0.5, label='test set %s' %bcur)
                #
                # fig.legend()
                # plt.title('train-test-splitted testing of %s' %bcur)
                # plt.savefig(filename, format='png')
                # print 'time %s, saving %s' %(str(datetime.now()), filename)

                del model
                del m
                gc.collect()
                os.chdir('../')
