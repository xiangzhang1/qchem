import random, string
from cStringIO import StringIO

from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

dynamic.global_load()

# any more-than-125 same-length from-scratch training produces 0.
# incremental training is no better.
# lowering learning rate seems to cause a significant improvement. presumably the idea is: the error goes to 225 because the calculation 'blows up' irreversibly. however, as soon as error goes to ~144, explosion happens.
# trying changing to linera activation does not end well. nor does extensively changing model structure.
#   tanh activation (all-LSTM or 1-LSTM): results in step function. presumably because summation is allowed.
#   elu and relu (all-LSTM) blows up.

curs = [
'master.PbS QD.bare qd testing.crunchit.3 opt',
'master.PbS QD.bare qd testing.crunchit.4 opt',
'master.PbS QD.bare qd testing.crunchit.5 opt',
'master.PbS QD.bare qd testing.crunchit.6 opt',
'master.PbS QD.bare qd testing.crunchit.7 opt',
'master.PbS QD.bare qd testing.crunchit.8 opt',
'master.PbS QD.bare qd testing.ml_opt.Pb48S48 rect',
'master.PbS QD.bare qd testing.ml_opt.Pb72S72 rect',
'master.PbS QD.bare qd testing.ml_opt.Pb36S36 rect'
]

m = dynamic.MlPbSOptXRNN()
for cur in curs:
    n = engine.Map().lookup(cur)
    m.parse_train(n.vasp)

m.train()
