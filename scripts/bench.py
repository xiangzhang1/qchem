# import engine
import numpy as np
import os
import time
import psutil

def factor1(n):
    """returns a list of prime factors of n"""
    d = 2
    factors = [ ]  #empty list
    while n > 1:
      if n % d == 0:
        factors.append(d)
        n = n/d
      else:
        d = d + 1
    return factors

n = engine.Map().lookup('master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.static0')

n.cell = engine.Cell('''Pb4 S4
1
6.01417 0.0 0.0
0.0 6.01417 0.0
0.0 0.0 6.01417
Pb S
4 4
Direct
0.0 0.0 0.0
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
0.5 0.0 0.0
0.5 0.5 0.5
0.0 0.0 0.5
0.0 0.5 0.0
'''
)

for ncore_node in [8, 12, 16, 18]:
    for ncore in set(factor1(ncore_node)):
        for kpar in set(factor1(ncore)):
            for prec in ['Accurate','Normal','Low']:
                for lreal in ['False','True']:

                    n.property = '''
                    eigenfunction, engine=vasp, spin=para,
                    ediff=1E-3, isym=2, ismear=-5,
                    ncore=2, ncore_node=16, prec=Accurate, encut=600, lreal=False, kpoints=G 24 24 24, kpar=8,
                    read=n, platform=dellpc
                    '''

                    n.reset()
                    n.compute()

                    with open("/home/xzhang1/m_cpu_config.log", "a") as of_:
                        of_.write('%s %s %s %s %s %s %s ' %(n.gen.memory['projector_real'], n.gen.memory['projector_reciprocal'], n.gen.getkw('npar'), n.gen.memory['wavefunction'], n.gen.getkw('kpar'), n.gen.memory['arraygrid'], int(n.gen.getkw('nnode')) * int(n.gen.getkw('ncore_node')) ) )
                    os.system('pkill vasp')
                    basic_mem = psutil.virtual_memory().used / 1000000.0
                    os.chdir(n.path)
                    os.system('./wrapper')
                    while True:
                        time.sleep(3)
                        with open("run.log","r") as if_:
                            if if_.readlines()[-1].startswith("DAV:"):
                                mem = psutil.virtual_memory().used / 1000000.0 - basic_mem
                                with open("/home/xzhang1/m_cpu_config.log", "a") as of_:
                                    of_.write('%s\n' %mem)
                                os.system('pkill vasp')
                                break
