#!/usr/bin/python
import numpy as np
import sys
import os
from pprint import pprint

from sklearn import preprocessing
from sklearn.svm import SVR

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = []
with open("m_cpu_config.log", "r") as if_:
    lines = if_.readlines()
    for line in lines:
        if line and not line.startswith('#') and len(line.split())==len(lines[1].split()):
            data.append( np.float_(line.split()) )
data = np.float_(data)




# training
MAX = 550
X_train = data[:MAX, :-3]
Y_train = data[:MAX, -1]
X_test = data[MAX:MAX+50, :-3]
Y_test = data[MAX:MAX+50, -1]
# Y_test_poly = data[MAX:MAX+50, -3]

# data = preprocessing.normalize(data, norm='l2', axis=0)
# data[:, (0,1,3,5)] /= 1000000000
# data[:, (-3, -2)] /= 1000

X_scaler = preprocessing.StandardScaler().fit(X_train)
X_train = X_scaler.transform(X_train)

Y_train = Y_train.reshape(-1, 1)
Y_scaler = preprocessing.StandardScaler().fit(Y_train)
Y_train = Y_scaler.transform(Y_train)

svr = SVR(kernel='poly',degree=3)
svr.fit(X_train, Y_train)




#testing
X_test = X_scaler.transform(X_test)
Y_test_pred = Y_scaler.inverse_transform(svr.predict(X_test))


#printing
text_to_print = zip(Y_test, Y_test_pred) #, Y_test_poly
np.savetxt(sys.stdout, text_to_print, fmt='%s', delimiter=' ')
