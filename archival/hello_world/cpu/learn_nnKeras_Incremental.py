#python
import numpy as np
import sys
import os
from pprint import pprint
from tqdm import tqdm

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ''

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from sklearn.neural_network import MLPRegressor

data = []
with open("m_cpu_config.log", "r") as if_:
    lines = if_.readlines()
    for line in lines:
        if line and not line.startswith('#') and len(line.split())==len(lines[1].split()):
            data.append( np.float_(line.split()) )
data = np.float_(data)

data[:, (0,2,3)] /= 1000000000 / 10.0  # wavefunction memory
data[:, (5)] /= 1000 / 10.0            # volume
data[:, (8, 9)] /= 1000         # memory used

MAX = 1000
TEST = 148
START = 8
EPOCH = 2
X_train = data[:MAX, :-3]
Y_train = data[:MAX, -1].T
X_test = data[MAX:MAX+TEST, :-3]
Y_test = data[MAX:MAX+TEST, -1].T
# Y_test_poly = data[MAX:MAX+TEST, -3]


# training
model_1 = Sequential([
    Dense(8, activation='relu', input_dim=8),
    Dropout(0.05),
    Dense(4, activation='relu'),
    Dropout(0.05),
    Dense(1, activation='linear')
])
model_2 = Sequential([
    Dense(8, activation='relu', input_dim=8),
    Dropout(0.05),
    Dense(4, activation='relu'),
    Dropout(0.05),
    Dense(1, activation='linear')
])
model_3 = Sequential([
    Dense(8, activation='relu', input_dim=8),
    Dropout(0.05),
    Dense(4, activation='relu'),
    Dropout(0.05),
    Dense(1, activation='linear')
])
model_1.compile(optimizer='rmsprop',
              loss='mse')
model_2.compile(optimizer='rmsprop',
              loss='mse')
model_3.compile(optimizer='rmsprop',
              loss='mse')
regr_1 = MLPRegressor(hidden_layer_sizes=(6, 4))
regr_2 = MLPRegressor(hidden_layer_sizes=(6, 4))


model_1.fit(X_train[:START], Y_train[:START], epochs=30, verbose=0) #, batch_size=32
model_2.fit(X_train[:START], Y_train[:START], epochs=30, verbose=0) #, batch_size=32
for _ in range(EPOCH):
    regr_1.fit(X_train[:START], Y_train[:START])

f = open('test.log','w')
for i in tqdm(range(START+1, MAX)):
    model_1.fit(np.array([X_train[i]]), np.array([Y_train[i]]), epochs=2, verbose=0)
    error_m1 = np.linalg.norm(model_1.predict(np.float_([X_train[i+1]])) - np.float_([Y_train[i+1]])) / Y_train[i+1]
    model_2.fit(X_train[:i], Y_train[:i], epochs=30, verbose=0)
    error_m2 = np.linalg.norm(model_2.predict(np.float_([X_train[i+1]])) - np.float_([Y_train[i+1]])) / Y_train[i+1]
    model_3.reset_states()
    model_3.fit(X_train[:i], Y_train[:i], epochs=30, verbose=0)
    error_m3 = np.linalg.norm(model_3.predict(np.float_([X_train[i+1]])) - np.float_([Y_train[i+1]])) / Y_train[i+1]
    for _ in range(EPOCH):
        regr_1.partial_fit([X_train[i]], [Y_train[i]])
        regr_2.fit(X_train[:i], Y_train[:i])
    error_r1 = np.linalg.norm(regr_1.predict([X_train[i+1]]) - [Y_train[i+1]]) / Y_train[i+1]
    error_r2 = np.linalg.norm(regr_2.predict([X_train[i+1]]) - [Y_train[i+1]]) / Y_train[i+1]
    f.write('%s %s %s %s %s %s\n' %(i, error_m1, error_m2, error_m3, error_r1, error_r2))
close(f)

# testing
# Y_test_pred = model.predict(X_test)
#
# text_to_print = zip(Y_test, Y_test_pred) #, Y_test_poly
# np.savetxt(sys.stdout, text_to_print, fmt='%s', delimiter=' ')
