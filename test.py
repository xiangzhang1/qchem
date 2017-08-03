import qchem
import shared
import engine

#for e in shared.ELEMENTS:
#    print e.symbol

#with open('data/markdown') as f:
#    qchem.Import(f.read())

with open('POSCAR','r') as f:
    engine.Cell(f.read())
