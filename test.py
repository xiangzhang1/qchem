dynamic.global_load()
m = dynamic.MlVaspSpeed()
for n in engine.Map().lookup('master').map.traverse():
    try:
        n.cell = engine.Cell(str(n.cell))
        n.gen.cell = n.cell
        n.vasp.cell = n.cell
        n.vasp.gen = n.gen
        n.vasp.optimized_cell = engine.Cell(str(n.vasp.optimized_cell))
    except AttributeError:
        pass
    if getattr(n, 'gen', None) and n.gen.parse_if('engine=vasp') and n.moonphase()==2:
        try:
            m.parse_obj(n.vasp, engine.Makeparam(n.vasp.gen))
        except (shared.CustomError, shared.DeferError) as e:
            print 'warning: node %s\'s parsing failed. probably old version.' %n.name
m.train()







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


dynamic.global_load()
m = dynamic.MlPbSOpt()
for n in engine.Map().lookup('master').map.traverse():
    try:
        m.parse_obj(n.vasp)
    except (shared.CustomError, shared.DeferError, AttributeError) as e:
        print 'warning: node %s\'s parsing failed. probably old version.' %n.name
m.train()
