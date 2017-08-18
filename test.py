import shared
import qchem
import engine
import subprocess


#with open('data/markdown') as f:
#    qchem.Import(f.read())

#with open('POSCAR','r') as f:
#    engine.Cell(f.read())

#qchem.Load()

#print shared.NODES['master'].map

#bqb = engine.Map().lookup('master.PbS QD.bare qd benchmarks')
#delattr(bqb, 'phase')

#qchem.Dump()


p = subprocess.Popen(['/home/xzhang1/src/qchem/qchem/test2.sh'], stdout=subprocess.PIPE, bufsize=1) 
for line in iter(p.stdout.readline, b''):
    print line,
p.stdout.close()
p.wait()
