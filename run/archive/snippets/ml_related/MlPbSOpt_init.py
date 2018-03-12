curs = [
# # Reproduce
#     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb108S108.g opt',
#     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb68S68.g opt',
#     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb63S62.g opt',
#     'master.PbS QD.bare qd testing.Q0 Reproduce.Pb55S38.g opt',
# # # Test convergence
#     'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02 | omg i forgot isym0',
#     'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02',
#     'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.10alt -0_01 opt',
#     'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.20alt -0_02 opt contd',
#     'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_01 opt',
#     # 'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.start - regular grid | end - -0_02',
# Crunchit
    ['master.PbS QD.bare qd testing.crunchit.3 opt', 'master.PbS QD.bare qd testing.crunchit.3 opt'],
    ['master.PbS QD.bare qd testing.crunchit.4 opt', 'master.PbS QD.bare qd testing.crunchit.4 opt'],
    ['master.PbS QD.bare qd testing.crunchit.5 opt', 'master.PbS QD.bare qd testing.crunchit.5 opt'],
    ['master.PbS QD.bare qd testing.crunchit.6 opt', 'master.PbS QD.bare qd testing.crunchit.6 opt'],
    ['master.PbS QD.bare qd testing.crunchit.7 opt', 'master.PbS QD.bare qd testing.crunchit.7 opt'],
]

shared.NODES = shared.load('NODES')
m = dynamic.MlPbSOpt()
for start_cur, cur in curs:
    n = engine.Map().lookup(cur)
    print 'parsing cur %s' %cur
    m.parse_train(n.vasp)
    print 'parsing complete.'

m.train()



# -----------------

for start_cur, end_cur in curs:
    start_ccoor = engine.Map().lookup(start_cur).cell.ccoor
    end_ccoor = engine.Map().lookup(end_cur).vasp.optimized_cell.ccoor
    print 'global shift is %s'%(np.sum(end_ccoor - start_ccoor, axis=0))

# -----------------


n = engine.Map().lookup('master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_02 opt')
m.parse_train(n.vasp)
for i in range(len(m._X)-93, len(m._X)):
    _X = m._X[i]
    _y0 = m._y0[i].reshape(-1)
    _y = m.predict(_X).reshape(-1)
    print _y0, _y-_y0
