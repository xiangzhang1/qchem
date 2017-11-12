dynamic.global_load()
m = dynamic.MlQueueTime()
for n in engine.Map().lookup('master').map.traverse():
    if getattr(n, 'vasp', None):
        try:
            m.parse_train(n, n.vasp, n.gen, n.cell, engine.Makeparam(n.gen))
        except (shared.CustomError, shared.DeferError) as e:
            print 'warning: node %s\'s parsing failed. probably old version.' %n.name ; sys.stdout.flush()
