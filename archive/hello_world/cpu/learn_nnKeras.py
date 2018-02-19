#!/usr/bin/env python
import numpy as np
import sys
import os
from pprint import pprint

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization

from sklearn.preprocessing import StandardScaler

data = []
with open("/home/xzhang1/m_cpu_config.log", "r") as if_:
    lines = if_.readlines()
    for line in lines:
        if line and not line.startswith('#') and len(line.split())==len(lines[1].split()):
            data.append( np.float_(line.split()) )
data = np.float_(data)

# data[:, (0,2,3)] /= 1000000000 / 10.0  # wavefunction memory
# data[:, (5)] /= 1000 / 10.0            # volume
# data[:, (8, 9)] /= 1000         # memory used


MAX = 1200
TEST = 148
X_train = data[:MAX, :-3]
Y_train = data[:MAX, -1].T
X_test = data[MAX:MAX+TEST, :-3]
Y_test = data[MAX:MAX+TEST, -1].T

# X_scaler = StandardScaler().fit(X_train)
# X_train = X_scaler.transform(X_train)

# training

model = Sequential([
    Dense(8, activation='relu', input_dim=8),
    Dropout(0.05),
    Dense(4, activation='relu'),
    Dropout(0.05),
    Dense(1, activation='linear')
])

model.compile(optimizer='rmsprop',
              loss='mse')

model.fit(X_train[:MAX], Y_train[:MAX], epochs=50, verbose=0) #, batch_size=32

# testing
Y_test_pred = model.predict(X_test)

text_to_print = zip(Y_test, Y_test_pred) #, Y_test_poly
np.savetxt(sys.stdout, text_to_print, fmt='%s', delimiter=' ')
