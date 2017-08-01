import qchem
import shared

#with open('data/markdown') as f:
#    qchem.Import(f.read())

#print shared.NODES['master']

#print shared.NODES['master'].map.lookup('master.qd_facet').name

#for i in shared.NODES['master'].map.lookup('master.qd_base.bulk').map._dict:
#    print i.name

x = qchem.Node('master\n\nmap:\n\n')
