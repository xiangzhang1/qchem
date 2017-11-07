#!/usr/bin/python
'''
An environment. Do not paste actions here.
'''

# common libraries
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import shutil
import random
import string
import dill as pickle
import time
from pprint import pprint
import IPython
import numpy as np

from cStringIO import StringIO
from fuzzywuzzy import process

import tensorflow as tf


# qchem package
import qchem
import shared
import dynamic
import engine
from shared import ELEMENTS

print shared.bcolors.OKBLUE + 'Welcome. Libraries loaded.' + shared.bcolors.ENDC


sys.stdout = open('test.py.log', 'w')



# ---------------------------









# dynamic.global_load()
# m = dynamic.MlVaspSpeed()
# for n in engine.Map().lookup('master').map.traverse():
#     try:
#         n.cell = engine.Cell(str(n.cell))
#         n.gen.cell = n.cell
#         n.vasp.cell = n.cell
#         n.vasp.gen = n.gen
#         n.vasp.optimized_cell = engine.Cell(str(n.vasp.optimized_cell))
#     except AttributeError:
#         pass
#     if getattr(n, 'gen', None) and n.gen.parse_if('engine=vasp') and n.moonphase()==2:
#         try:
#             m.parse_obj(n.vasp, engine.Makeparam(n.vasp.gen))
#         except (shared.CustomError, shared.DeferError) as e:
#             print 'warning: node %s\'s parsing failed. probably old version.' %n.name
# m.train()







# -------------------------------------------------------------




# import numpy as np
# import tensorflow as tf
#
# path_X = '/home/xzhang1/data_X.npy'
# path_y0 = '/home/xzhang1/data_y0.npy'
# _X = np.load(path_X)
# _y0 = np.load(path_y0)
# _y0 = np.log(_y0)
#
# tf.reset_default_graph()
#
# X = tf.placeholder(tf.float32, shape=(None, 12))
# y0 = tf.placeholder(tf.float32, shape=(None, 1))
#
# y1 = tf.layers.dense(X, units=8, activation=tf.nn.relu)
# y2 = tf.layers.dense(y1, units=3, activation=tf.nn.relu)
# y3 = tf.layers.dense(y1, units=3, activation=tf.nn.relu)
# y = tf.layers.dense(y3, units=1)
#
# loss = tf.nn.l2_loss(y - y0)
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# training_op = optimizer.minimize(loss)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for _ in range(1000):
#         _y, _loss, tmp = sess.run([y, loss, training_op], feed_dict={X: _X, y0: _y0})
#         if _ % 100 == 0:
#             print 'Epoch %s, actual %s, predicted %s, loss %s' %(_, _y0, _y, _loss)





# -------------------------------------------------------------

curs = [
# Reproduce
    'master.PbS QD.bare qd testing.Q0 Reproduce.Pb108S108.g opt',
    'master.PbS QD.bare qd testing.Q0 Reproduce.Pb68S68.g opt',
    'master.PbS QD.bare qd testing.Q0 Reproduce.Pb63S62.g opt',
    'master.PbS QD.bare qd testing.Q0 Reproduce.Pb55S38.g opt',
# Test convergence
    # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0',
    # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02',
    # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed | end - -0_02',
    # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed #2 | end - -0_02',
# Crunchit
    'master.PbS QD.bare qd testing.crunchit.3 opt',
    'master.PbS QD.bare qd testing.crunchit.4 opt',
    'master.PbS QD.bare qd testing.crunchit.5 opt',
    'master.PbS QD.bare qd testing.crunchit.6 opt',
    'master.PbS QD.bare qd testing.crunchit.7 opt',
]

dynamic.global_load()
m = dynamic.MlPbSOpt()
# m = dynamic.load('MlPbSOpt')
# X_path = '/home/xzhang1/X.npy'
# y_path = '/home/xzhang1/y.npy'
# np.save(X_path, m._X)
# np.save(y_path, m._y0)
# m._X = np.load(X_path)
# m._y0 = np.load(y_path)
# for n in engine.Map().lookup('master').map.traverse():
#     if getattr(n, 'vasp', None) and getattr(n.vasp, 'optimized_cell', None):
#         m.parse_obj(n.vasp)
for cur in curs:
    n = engine.Map().lookup(cur)
    print 'parsing cur %s' %cur
    m.parse_obj(n.vasp)
    print 'parsing complete.'
dynamic.save(m, m.__class__.__name__)
m.train()
