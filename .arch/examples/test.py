import qchem
import shared

with open('data/markdown') as f:
    qchem.Import(f.read())

print dynamic.NODES['master']

#print dynamic.NODES['master'].map.lookup('master.qd_facet').name

#for i in dynamic.NODES['master'].map.lookup('master.qd_base.bulk').map._dict:
#    print i.name

