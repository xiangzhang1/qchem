import qchem

with open('data/markdown') as f:
    qchem.Import(f.read())

#qchem.Dump()

qchem.Load()

print qchem.NODES
