#python
import numpy as np
import sys
import os
from pprint import pprint

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

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

MAX = 1200
TEST = 148
X_train = data[:MAX, :-3]
Y_train = data[:MAX, -1].T
X_test = data[MAX:MAX+TEST, :-3]
Y_test = data[MAX:MAX+TEST, -1].T
# Y_test_poly = data[MAX:MAX+TEST, -3]


# training
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4))
regr_1.fit(X_train, Y_train)
regr_2.fit(X_train, Y_train)

# testing
Y_test_pred1 = regr_1.predict(X_test)
Y_test_pred2 = regr_2.predict(X_test)

text_to_print = zip(Y_test_pred1, Y_test_pred2) #, Y_test_poly
np.savetxt(sys.stdout, text_to_print, fmt='%s', delimiter=' ')
