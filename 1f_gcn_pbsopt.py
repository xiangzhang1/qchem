#!/usr/bin/env python
import dynamic
import engine
import qchem

import tensorflow as tf
import IPython
import numpy as np
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer, StandardScaler
import dill as pickle


dynamic.global_load()

curs = [
    # ['master.PbS QD.bare qd testing.crunchit.3 opt','master.PbS QD.bare qd testing.crunchit.3 opt'],
    # ['master.PbS QD.bare qd testing.crunchit.4 opt','master.PbS QD.bare qd testing.crunchit.4 opt'],
    # ['master.PbS QD.bare qd testing.crunchit.5 opt','master.PbS QD.bare qd testing.crunchit.5 opt'],
    # ['master.PbS QD.bare qd testing.crunchit.6 opt','master.PbS QD.bare qd testing.crunchit.6 opt'],
    # ['master.PbS QD.bare qd testing.crunchit.7 opt','master.PbS QD.bare qd testing.crunchit.7 opt'],
    ['master.PbS QD.bare qd testing.crunchit.8 opt half-complete','master.PbS QD.bare qd testing.crunchit.8 opt0'],
    # ['master.PbS QD.bare qd testing.ml_opt.Pb48S48 rect','master.PbS QD.bare qd testing.ml_opt.Pb48S48 rect'],
    # ['master.PbS QD.bare qd testing.ml_opt.Pb72S72 rect','master.PbS QD.bare qd testing.ml_opt.Pb72S72 rect'],
    # ['master.PbS QD.bare qd testing.ml_opt.Pb36S36 rect','master.PbS QD.bare qd testing.ml_opt.Pb36S36 rect'],
    # ['master.PbS QD.bare qd testing.ml_opt.Pb40S40 customshape','master.PbS QD.bare qd testing.ml_opt.Pb40S40 customshape'],
    # ['master.PbS QD.bare qd testing.ml_opt.Pb80S80 1 13','master.PbS QD.bare qd testing.ml_opt.Pb80S80 1 13'],
    # ['master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0','master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0']
]




FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('summaries-dir', './GCN', """Directory where to write loss history and checkpoint""")
tf.app.flags.DEFINE_integer('epochs', 5000, """How many mini-epochs to run""")
tf.app.flags.DEFINE_integer('steps-per-epoch', 10, """How many optimizer.step() to run per mini-epoch""")
tf.app.flags.DEFINE_integer('conv-steps', 5, """How many graph-convolution steps to run""")
tf.app.flags.DEFINE_float('learning-rate', 0.01, """Learning rate for minimizer""")
tf.app.flags.DEFINE_string('optimizer', 'RMSPropOptimizer', """RMSPropOptimizer | AdamOptimizer""")




_As = []
_Xs = []
_y0s = []
for bcur, ecur in curs:

    bn = engine.Map().lookup(bcur)
    bcell = bn.cell
    bcoor = bn.cell.ccoor
    en = engine.Map().lookup(ecur)
    ecoor = en.vasp.optimized_cell.ccoor
    natom0 = en.cell.stoichiometry.values()[0]
    natom = en.cell.natoms()
    a = 6.014 / 2

    _A = np.zeros((7, natom, natom))    # 0, x+, x-, y+, y-, z+, z-

    bottomLeftCorner = np.min(bcoor, axis=0)
    rcoor = np.int_(np.around((bcoor - bottomLeftCorner) / a))

    _A[0, :, :] = np.eye(natom)
    for i, rc in enumerate(rcoor):
        sgn = np.sign((i - natom0 + 0.5) * (np.arange(natom) - natom0 + 0.5))

        for idisplace, displace in enumerate([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]):
            temp = np.where((rcoor == rc + displace).all(axis=1))
            if temp and temp[0]:
                _A[idisplace + 1, i, temp[0][0]] = sgn[temp[0][0]]

    _X = np.zeros((natom, 4))
    # Please, try _X[:, 3] = np.sign(np.arange(natom) - natom0 + 0.5) later as well. Maybe try cell as well. Or _X=0
    _X[:, 3] = np.sign(np.arange(natom) - natom0 + 0.5)
    _X[:, :3] = bcoor

    _y0 = (ecoor - bcoor)[:,0:1] * 100

    _As.append(_A)
    _Xs.append(_X)
    _y0s.append(_y0)




A = tf.placeholder(tf.float32, [7, None, None])
X = tf.placeholder(tf.float32, [None, 4])
W = tf.Variable(tf.random_normal([4, 4, 7]))
b = tf.Variable(tf.random_normal([4, 7]))
H = X   # I need X's handle.

for _ in range(FLAGS.conv_steps):
    HW = tf.tensordot(H, W, axes=[[1], [0]])
    HWb = tf.add(HW, b)
    AHWb = tf.tensordot(A, HWb, axes=[[2,0],[0,2]])
    H = tf.nn.relu(AHWb)

if_HN =
HN =

H = tf.layers.dense(
    inputs=H,
    units=16,
    activation=tf.nn.relu
)

H = tf.layers.dense(
    inputs=H,
    units=8,
    activation=tf.nn.relu
)

y = tf.layers.dense(
    inputs=H,
    units=1,
    activation=None
)

y0 = tf.placeholder(tf.float32, [None, 1])




loss = tf.reduce_mean((y0 - y) ** 2)

relative_loss = tf.sqrt(loss) / tf.sqrt(tf.reduce_mean(y0 ** 2))

train_step = getattr(tf.train, FLAGS.optimizer)(FLAGS.learning_rate).minimize(loss)

summary = tf.summary.scalar('relative loss', relative_loss)

saver = tf.train.Saver()

sess = tf.InteractiveSession()

train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train')

test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

# tf.global_variables_initializer().run()
saver.restore(sess, FLAGS.summaries_dir + '/model.ckpt')

for epoch, i in enumerate(np.random.choice(len(_As), FLAGS.epochs)):

    if epoch % 100 == 0:    # test
        _y, _relaive_loss, _summary = sess.run([y, relative_loss, summary], feed_dict={A: _As[i], X: _Xs[i], y0: _y0s[i]})
        test_writer.add_summary(_summary, epoch)
        with open(FLAGS.summaries_dir + '/out.pickle', 'wb') as f:
            pickle.dump((_y0, _y), f)
        saver.save(sess, FLAGS.summaries_dir + '/model.ckpt')

    else:   # train
        for it in range(FLAGS.steps_per_epoch):
            _relative_loss, _summary, _ = sess.run([relative_loss, summary, train_step], feed_dict={A: _As[i], X: _Xs[i], y0: _y0s[i]})
            if it % 1000 == 0:
                print 'Relative loss at epoch %s / %s, iteration %s / %s: %s' % (epoch, FLAGS.epochs, it, FLAGS.steps_per_epoch, _relative_loss)
        train_writer.add_summary(_summary, epoch)
        print 'Relative loss at epoch %s / %s: %s' % (epoch, FLAGS.epochs, _relative_loss)

sess.close()
