dynamic.global_load()
m = dynamic.MlQueueTime()
for n in engine.Map().lookup('master').map.traverse():
    if getattr(n, 'vasp', None):
        m.parse_train(n, n.vasp, n.gen, n.cell, engine.Makeparam(n.gen))
