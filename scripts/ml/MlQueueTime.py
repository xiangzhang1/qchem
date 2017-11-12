dynamic.global_load()
m = dynamic.MlQueueTime()
for n in engine.Map().lookup('master').map.traverse():
    if getattr(n, 'vasp', None):
        print 'parsing %s' %n.name
        m.parse_train(n, n.gen, n.vasp)
