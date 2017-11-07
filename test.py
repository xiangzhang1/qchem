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
        except shared.CustomError:
            print 'warning: node %s\'s parsing failed. probably old version.' %n.name
dynamic.global_save()
m.train()
