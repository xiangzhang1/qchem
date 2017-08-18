import shared
import qchem
import engine


#with open('data/markdown') as f:
#    qchem.Import(f.read())

#with open('POSCAR','r') as f:
#    engine.Cell(f.read())

qchem.Load()

print shared.NODES['master'].map

bqb = engine.Map().lookup('master.PbS QD.bare qd benchmarks')
delattr(bqb, 'phase')

qchem.Dump()
