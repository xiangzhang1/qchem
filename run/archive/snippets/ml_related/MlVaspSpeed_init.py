shared.nodes = shared.load('nodes')
m = dynamic.MlVaspSpeed()
for n in engine.Map().lookup('master').map.traverse():
    if getattr(n, 'vasp', None):
        try:
            m.parse_train(n, n.vasp, n.gen, n.cell, engine.Makeparam(n.gen))
        except (shared.CustomError, shared.DeferError) as e:
            print 'warning: node %s\'s parsing failed. probably old version.' %n.name ; sys.stdout.flush()



def f(x, m=m, optimizer_name='SGD'):    #  train then return error. for hyperparameter search.
    print '----------------------------' ; sys.stdout.flush()
    x = abs(x)
    bn_momentum, dropout_p, learning_rate, batch_size, n_epochs = x[0] / 10.0, x[1] / 15.0, 10**(-1*x[2]), int(10 * x[3]), int(1000 * x[4])
    m.net = dynamic.MlVaspSpeedNet(bn_momentum=bn_momentum, dropout_p=dropout_p)
    err = m.train(learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs, optimizer_name=optimizer_name)
    print 'parameters: %s. error: %s.' %(x, err) ; sys.stdout.flush()
    return err

from scipy.optimize import minimize
print minimize(f, x0=np.float32([9, 1, 2, 3.2, 4]), method='Powell') ; sys.stdout.flush()
#  9.704  1.159  1.591  6.478  5.784   |     0.97, 0.077, 0.026, 64, 5800
print 'finished! :)'
