#!/usr/bin/python

curs = [
# # Reproduce
#     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb108S108.g opt',
#     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb68S68.g opt',
#     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb63S62.g opt',
#     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb55S38.g opt',
# # Test convergence
#     'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0',
#     # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02',
#     # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed | end - -0_02',
#     # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed #2 | end - -0_02',
# # Crunchit
#     'master.PbS QD.bare qd testing.crunchit.3 opt',
#     'master.PbS QD.bare qd testing.crunchit.4 opt',
#     'master.PbS QD.bare qd testing.crunchit.5 opt',
#     'master.PbS QD.bare qd testing.crunchit.6 opt',
    'master.PbS QD.bare qd testing.crunchit.7 opt',
]

dynamic.global_load()
m = dynamic.MlPbSOpt()
for cur in curs:
    n = engine.Map().lookup(cur)
    print 'parsing cur %s' %cur
    m.parse_obj(n.vasp)
    print 'parsing complete.'
dynamic.save(m, m.__class__.__name__)
m.train()

# ------------------------------------------------------------------------------

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
