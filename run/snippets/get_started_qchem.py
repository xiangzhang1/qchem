p = engine.Map().lookup('master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38')
p.map.add_edge('20alt -0_02 opt contd', 'compare cell')
n = p.map.lookup('compare cell')

n.property = 'engine=md, movie, elev=10, angle=1'
n.reset()
n.compute()
