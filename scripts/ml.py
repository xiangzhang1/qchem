#!/usr/bin/python
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
from pprint import pprint

data = []
with open("m_cpu_config.log", "r") as if_:
    lines = if_.readlines()
    for line in lines:
        if line and not line.startswith('#') and len(line.split())==len(lines[1].split()):
            data.append( np.float_(line.split()) )
data = np.float_(data)


data = preprocessing.normalize(data, norm='l2', axis=0)
scalar = preprocessing.StandardScaler().fit(data)
data=scalar.transform(data)

svr = SVR(kernel='poly',degree=3)


print '-' * 100

#train
svr.fit(data[:300,:-3],data[:300,-2])

#testing
predictions=svr.predict(data[300:,:-3])

for o,p,d in zip(data[300:,-2], predictions, data[300:,-3]):
   print d, o, p

print '-' * 100

#train
svr.fit(data[:300,:-3],data[:300,-1])

#testing
predictions=svr.predict(data[300:,:-3])

for p,d in zip(predictions, data[300:,-1]):
   print d, p
