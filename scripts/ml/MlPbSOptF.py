curs = [
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.opt ediffg -0_01',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.opt ediffg -0_03',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.opt ediffg -0_02_responsible',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.opt ediffg 1E-4',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.opt ediffg 1E-5',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.opt ediffg 1E-3',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.20alt -0_02 opt',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_02 opt arbitrary start 1',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_01 ibrion3 opt',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_01 ibrion1 opt - oscillate',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_01 ibrion3 opt_1',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_01 ibrion3 opt_2',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_01 ibrion3 opt_2_1',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_02 opt arbitrary start 2',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_01 ibrion1 opt2',
    'master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.15alt -0_01 ibrion3 opt2'
]

dynamic.global_load()
m = dynamic.MlPbSOptF()
for cur in curs:
    n = engine.Map().lookup(cur)
    print 'parsing cur %s' %cur
    m.parse_train(n.vasp)
    print 'parsing complete.'

_y0 = np.copy(m._y0)
m._X = np.delete(m._X, np.where([np.linalg.norm(c)<0.1 for c in _y0])[0], axis=0)
m._y0 = np.delete(m._y0, np.where([np.linalg.norm(c)<0.1 for c in _y0])[0], axis=0)

m.train()
