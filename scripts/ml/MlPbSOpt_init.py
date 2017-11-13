curs = [
# Reproduce
    'master.PbS QD.bare qd testing.Q0 Reproduce.Pb108S108.g opt',
    'master.PbS QD.bare qd testing.Q0 Reproduce.Pb68S68.g opt',
    'master.PbS QD.bare qd testing.Q0 Reproduce.Pb63S62.g opt',
    'master.PbS QD.bare qd testing.Q0 Reproduce.Pb55S38.g opt',
# # Test convergence
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.10alt -0_01 opt',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.20alt -0_02 opt contd',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_01 opt',
#     # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02',
#     # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed | end - -0_02',
#     # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - perturbed #2 | end - -0_02',
# Crunchit
    'master.PbS QD.bare qd testing.crunchit.3 opt',
    'master.PbS QD.bare qd testing.crunchit.4 opt',
    'master.PbS QD.bare qd testing.crunchit.5 opt',
    'master.PbS QD.bare qd testing.crunchit.6 opt',
    'master.PbS QD.bare qd testing.crunchit.7 opt',
]

dynamic.global_load()
m = dynamic.MlPbSOpt()
for cur in curs:
    n = engine.Map().lookup(cur)
    print 'parsing cur %s' %cur
    m.parse_train(n.vasp)
    print 'parsing complete.'

m.train()



# ------------------------------------------------------------------------------------------------


dynamic.global_load()
m = dynamic.MlPbSOpt()
for n in engine.Map().lookup('master').map.traverse():
    if getattr(n, 'vasp', None) and getattr(n, 'gen', None) and n.gen.parse_if('opt') and int(n.gen.getkw('nsw')) > n.vasp.info('n_ionic_step') and getattr(n.vasp, 'optimized_cell', None) and n.gen.parse_if('pbs & qd') and n.gen.parse_if('ibrion=2'):
        print 'parsing %s' %n.name
        try:
            m.parse_train(n.vasp)
        except (shared.CustomError, shared.DeferError, StopIteration) as e:
            print 'warning: node %s\'s parsing failed. probably old version.' %n.name



# -----------------


n = engine.Map().lookup('master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_02 opt')
m.parse_train(n.vasp)
for i in range(len(m._X)-93, len(m._X)):
    _X = m._X[i]
    _y0 = m._y0[i].reshape(-1)
    _y = m.predict(_X).reshape(-1)
    print _y0, _y-_y0
