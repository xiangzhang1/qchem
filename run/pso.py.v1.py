
# coding: utf-8

# In[80]:


# Numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.stats

# Machine learning
import tensorflow as tf
import keras
import sklearn
import sklearn.preprocessing, sklearn.base, sklearn.utils, sklearn.model_selection, sklearn.gaussian_process

import optunity

# Various Python tricks and libraries
import requests
import time
import functools
import operator
import collections
from tqdm import tqdm, tqdm_notebook, tnrange
import dill as pickle
import IPython
import gc

# Parallel
import joblib
import multiprocessing


# In[2]:


def read_log():
    jobs = pd.read_csv(filepath_or_buffer='ml_queue.log', sep='\s+', header='infer', skiprows=[1], na_values=['UNLIMITED','Unknown','kijana','root']).dropna()

    jobs.loc[:, 'Submit'] = pd.to_datetime(jobs.loc[:, 'Submit'].copy(), errors='coerce')
    jobs.loc[:, 'Start'] = pd.to_datetime(jobs.loc[:, 'Start'].copy(), errors='coerce')
    jobs.loc[:, 'End'] = pd.to_datetime(jobs.loc[:, 'End'].copy(), errors='coerce')

    jobs.loc[:, 'NNodes'] = pd.to_numeric(jobs.loc[:, 'NNodes'].copy(), errors='coerce', downcast='integer')

    jobs.loc[:, 'Timelimit'] = jobs.loc[:, 'Timelimit'].copy().str.replace('-','day ')
    jobs.loc[:, 'Timelimit'] = pd.to_timedelta(jobs.loc[:, 'Timelimit'].copy(), errors='coerce')
    
    jobs.loc[:, 'Waited'] = (jobs.Start - jobs.Submit).values / pd.Timedelta('1h')

#     jobs = jobs[(np.abs(scipy.stats.zscore(jobs.loc[:,'Wait'].values / pd.Timedelta('1h'))) <3)]
    jobs = jobs.dropna()
    
    jobs = jobs.sort_values(by = 'Submit')
    
    return jobs


# In[3]:


jobs = read_log()


# In[ ]:


def sample(tuple_):
    
    jobs, index, row = tuple_
    
    tj = thisjob = row
    now = tj.Submit
    
    rj = relatedjobs = jobs.loc[np.logical_and.reduce([jobs.index != index, jobs.Submit <= now, jobs.End > now])] # excludes thisjob, includes jobs submitted simultaneously but ranked earlier
    
    # 无关人等滚开
    if now < jobs.End.min() or index < 100 or index == 10826:
        return None
        
    # 零、有空位，就不用等。
    if rj.NNodes.sum() + tj.NNodes <= 29 and np.sum(rj.User == tj.User)<10 and rj.loc[rj.User == tj.User].NNodes.sum() + tj.NNodes<16:
        assert tj.Waited <= 0.3
        
    # 一、各种descriptor
    analysis = pd.DataFrame([], index=[index])
    
    analysis.loc[:, 'NNodes'] = tj.NNodes
    analysis.loc[:, 'ETA']    = tj.Timelimit
    
    analysis.loc[:, 'Total_squeue_nnodes'] = rj.NNodes.sum()
    
    analysis.loc[:, 'Total_squeue_eta']    = rj.Timelimit.sum()
    
    analysis.loc[:, 'Self_job_in_squeue_percentage'] = 0 if len(rj)==0 else float(np.sum(rj.User == tj.User)) / len(rj)
    
    analysis.loc[:, 'Waited'] = tj.Waited
    
    return analysis
    
    
L = multiprocessing.Pool(processes=20).map(sample, [(jobs, index, row.copy()) for index, row in jobs.iterrows()])
analyses = pd.concat(L) # None is automatically ignored


# In[42]:


normalize_analyses = pd.DataFrame(sklearn.preprocessing.scale(analyses.apply(pd.to_numeric).values), index=analyses.index, columns=analyses.columns)


# In[54]:


def r2_score(ytrue, ypred): # sklearn.metrics.r2_score in tensorflow. 1 output only. 

    ytrue_mean = tf.reduce_mean(ytrue, name="ytrue_mean")
    r2_score = tf.subtract(1., tf.truediv(tf.reduce_mean((ytrue - ypred) ** 2), tf.reduce_mean((ytrue - ytrue_mean) ** 2)), name="r2_score")
    return r2_score

class TqdmProgBar(keras.callbacks.Callback):
    '''features:
    1. tqdm ETA bar
    2. logs[field] plotted for each field in fields
    '''
    
    def __init__(self, n_epochs, fields, interval=10): 
        self.n_epochs = n_epochs
        self.fields = fields
        self.interval = interval
        
        self.fields_history = dict((field, []) for field in fields)
        self.fig, self.ax = plt.subplots(1, 1)
        
    def on_train_begin(self, logs):
        self.pbar = tqdm_notebook(total=self.n_epochs, leave=False)
        
    def on_train_end(self, logs):
        self.pbar.close()
        
    def on_epoch_end(self, epoch, logs):
        if epoch % self.interval == 0:
            self.pbar.update(self.interval)

            for field in self.fields:
                self.fields_history[field].append(logs[field])

            self.ax.clear()
            for field in self.fields:
                self.ax.plot(self.fields_history[field], label="%s=%.2f" %(field, self.fields_history[field][-1]))
            self.ax.legend(loc='best')
            self.fig.canvas.draw()
        


# In[117]:


# 'on1'='units1'. what a fucked-up grammar. still, it works: that variable essentially doesn't exist.
# or maybe they disabled nested categorical. oh well they fucking didn't.

search = {
    'activation': {
        'tanh': None,
        'elu': None
    },
    'dropout': [0, 1],
    'momentum': [0.5, 1],
    'optimizer': {
        'sgd': None,
        'rmsprop': None,
        'adam': None
    },
    'minus_log_lr': [0, 5],
    'log2_batch_size': [7, 13],
    'epochs': [200, 400],
    'layer1': {
        'off1': None,
        'on1': {
            'units1': [1, 16],
            'layer2': {
                'off2': None,
                'on2': {
                    'units2': [1, 16],
                    'layer3': {
                        'off3': None,
                        'on3': {
                            'units3': [1, 16],
                            'layer4': {
                                'off4': None,
                                'on4': {
                                    'units4': [1, 16],
                                    'layer5': {
                                        'off5': None,
                                        'on5': {
                                            'units5': [1, 16],
                                            'layer9': {
                                                'off6': None,
                                                'on6': {
                                                    'units6': [1, 16]
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


# In[118]:


def r2(normalize_analyses=normalize_analyses, activation=None, dropout=None, momentum=None, optimizer=None, minus_log_lr=None, log2_batch_size=None, epochs=None,
       layer1=None, units1=None, 
       layer2=None, units2=None, 
       layer3=None, units3=None, 
       layer4=None, units4=None, 
       layer5=None, units5=None, 
       layer6=None, units6=None, 
       *args, **kwargs
      ):
    
    lr = 10. ** -minus_log_lr
    batch_size = int(round(2. ** log2_batch_size))
    epochs = int(round(epochs))
    if units1: 
        units1 = int(units1)
        if units2:
            units2 = int(units2)
            if units3:
                units3 = int(units3)
                if units4:
                    units4 = int(units4)
                    if units5:
                        units5 = int(units5)
                        if units6:
                            units6 = int(units6)
    
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(5,))
    ])
    
    def add_to_model(units, model=model, activation=activation, momentum=momentum, dropout=dropout):
        model.add(keras.layers.Dense(units=units, activation=None))
        model.add(keras.layers.BatchNormalization(momentum=momentum))
        model.add(keras.layers.Activation(activation))
        model.add(keras.layers.Dropout(rate=dropout))    
    
    if layer1 == 'off1':
        return 0
    else:
        add_to_model(units1)
        if layer2 == 'off2':
            pass
        else:
            add_to_model(units2)
            if layer3 == 'off3':
                pass
            else:
                add_to_model(units3)
                if layer4 == 'off4':
                    pass
                else:
                    add_to_model(units4)
                    if layer5 == 'off5':
                        pass
                    else:
                        add_to_model(units5)
                        if layer6 == 'off6':
                            pass
                        else:
                            add_to_model(units6)
                            
    add_to_model(1, activation=None)
    model.compile(loss='mse', optimizer=optimizer, metrics=[r2_score])
    history = model.fit(normalize_analyses.drop('Waited', axis=1), normalize_analyses.Waited, validation_split=0.2, batch_size=batch_size, epochs=epochs, verbose=0)
    val_r2_score = history.history['val_r2_score'][-1]
    if np.isnan(val_r2_score):
        val_r2_score = -10
    train_r2_score = history.history['r2_score'][-1]
    if np.isnan(train_r2_score):
        train_r2_score = -10
        
    del model
    gc.collect()
    
    with open('pso.csv', 'a') as f:
        f.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n'%(activation, dropout, momentum, optimizer, lr, batch_size, epochs, units1, units2, units3, units4, units5, units6, train_r2_score, val_r2_score))

    return val_r2_score


# In[ ]:


with open('pso.csv', 'w') as f:
    f.write('activation, dropout, momentum, optimizer, lr, batch_size, epochs, layer1, layer2, layer3, layer4, layer5, layer6, train_r2, val_r2\n')
result = optunity.maximize_structured(r2, search_space=search, num_evals=1000)
with open('pso_result.pickle', 'wb') as f:
    pickle.dump(result, f)


# Due to the stake here, we copied the code to a standalone python file and executed it.
