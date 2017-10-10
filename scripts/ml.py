import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
from pprint import pprint

data = []
with open("m_cpu_config.log", "r") as if_:
    for line in if_.readlines():
        if line and not line.startswith('#'):
            data.append( np.float_(line.split()) )
data = np.float_(data)
print data.shape

#x1=np.random.randint(1000,size=(1000,1))
#x2=np.random.randint(1000,size=(1000,1))
#x3=np.random.randint(10000,size=(1000,1))
#x4=np.random.randint(10,size=(1000,1))
#x5=np.random.randint(10,size=(1000,1))
#y=(x1+x2)*x4 +(x3*x5)+750
#data=np.concatenate((x1,x2,x3,x4,x5,y),axis=1)

scalar = preprocessing.StandardScaler().fit(data)
data_normalized=scalar.transform(data)

svr = SVR(kernel='poly',degree=3)

#800 train, 200:test
svr.fit(data[:,:-1],data[:,-1])

#testing
predictions=svr.predict(data[:,:-1])

print '-' * 100
for p,l in zip(predictions,data[:,-1]):
   print (p,l)
