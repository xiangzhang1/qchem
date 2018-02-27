
# coding: utf-8

# # My Function Zoo

# This notebook curates a list of helper functions. When a project finishes, reusable functions are moved here.

# In[ ]:


# Numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine learning
import tensorflow as tf
import keras
import sklearn
import sklearn.preprocessing, sklearn.base, sklearn.utils

# Various Python tricks and libraries
import requests
import time
import functools
import operator
import collections


# ## Python

# ### Iterable to list

# In[ ]:


def is_iterable(L):
    return hasattr(L, '__iter__')

def to_iterable(L):
    if isinstance(L, (np.ndarray, np.generic)):
        return L.tolist()
    if isinstance(L, pd.DataFrame):
        return L.values.tolist()
    if is_iterable(L):
        return recursive_map(lambda x:x, L)
    raise ValueError
    
def flatten(L):
    return reduce(operator.add, map(lambda l: flatten(l) if is_iterable(l) else [l], L))

def recursive_map(func, L):
    return map(lambda l: recursive_map(func, l) if is_iterable(l) else func(l), L)

def get_index(item, L, index_unexpected=-1, random_unexpected=0.): # first occurence or -1
    return index_unexpected if np.random.rand() < random_unexpected or item not in L else np.argmax(np.array(L)==item)

def get_value(index, L, value_unexpected_index=None, index_unexpected=-1): # L[index] or None
    return value_unexpected_index if index == index_unexpected else L[index]


# ## Machine Learning

# ### Categorical Data

# In[ ]:


class LabelEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin): 
    # sklearn.preprocessing.LabelEncoder with irregular 2D array, unexpected index or value, and random -1 return.
    
    def __init__(self, index_unexpected=-1, random_unexpected=0., value_unexpected_index=None):
        self.index_unexpected = index_unexpected
        self.random_unexpected = random_unexpected
        self.value_unexpected_index = value_unexpected_index
        
    def fit(self, y):
        y_flattened = flatten(y)
        self.classes_ = np.unique(y_flattened)
        return self

    def transform(self, y):
        func = functools.partial(get_index, 
                                 L=self.classes_, 
                                 index_unexpected=self.index_unexpected, 
                                 random_unexpected=self.random_unexpected)
        return recursive_map(
            func=lambda item: get_index(
                item=item, 
                L=self.classes_, 
                index_unexpected=self.index_unexpected, 
                random_unexpected=self.random_unexpected),
            L=y)

    def inverse_transform(self, y):
        return recursive_map(
            func=lambda index: get_value(
                index=index, 
                L=self.classes_, 
                index_unexpected=self.index_unexpected, 
                random_unexpected=self.random_unexpected),
            L=y)
    

