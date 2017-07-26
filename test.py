import qchem
import shared

with open('data/markdown') as f:
    qchem.Import(f.read())

print shared.NODES['master'].map

print shared.NODES['master'].map.lookup('master.qd_base').name
