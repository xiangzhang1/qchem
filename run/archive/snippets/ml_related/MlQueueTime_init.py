shared.nodes = shared.load('nodes')
m = dynamic.MlQueueTime()
for n in engine.Map().lookup('master').map.traverse():
    if getattr(n, 'vasp', None):
        print 'parsing %s' %n.name
        try:
            m.parse_train(n, n.gen, n.vasp)
        except (shared.IllDefinedError, shared.MaybeTemporaryError, StopIteration) as e:
            print 'warning: node %s\'s parsing failed. probably old version.' %n.name ; sys.stdout.flush()
