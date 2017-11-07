import numpy as np
import tensorflow as tf

path_X = '/home/xzhang1/data_X.npy'
path_y0 = '/home/xzhang1/data_y0.npy'
_X = np.load(path_X)
_y0 = np.load(path_y0) / 10

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, 16))
y0 = tf.placeholder(tf.float32, shape=(None, 1))

y1 = tf.layers.dense(X, units=8, activation=tf.nn.relu)
# y2 = tf.layers.dense(y1, units=3, activation=tf.nn.relu)
y3 = tf.layers.dense(y1, units=3, activation=tf.nn.relu)
y = tf.layers.dense(y3, units=1)

loss = tf.nn.l2_loss(y - y0)
optimizer = tf.train.GradientDescentOptimizer(0.01)
training_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        _y, _loss, tmp = sess.run([y, loss, training_op], feed_dict={X: _X, y0: _y0})
        if _ % 100 == 0:
            print 'Epoch %s, actual %s, predicted %s, loss %s' %(_, _y0, _y, _loss)
