# import engine
import numpy as np
import os
import time
import psutil

def factor1(n):
    """returns a list of prime factors of n"""
    factors = []
    for i in range(1,n+1):
        if n % i == 0:
            factors.append(i)
    return factors

n = engine.Map().lookup('master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.static0')

n.cell = engine.Cell('''Pb S
1
29.0213	0.0	0.0
0.0	29.0213	0.0
0.0	0.0	29.0213
Pb  S
32  32
Direct
0.344575 0.344575 0.344575
0.344575 0.448192 0.448192
0.448192 0.344575 0.448192
0.448192 0.448192 0.344575
0.344575 0.344575 0.551808
0.344575 0.448192 0.655425
0.448192 0.344575 0.655425
0.448192 0.448192 0.551808
0.344575 0.551808 0.344575
0.344575 0.655425 0.448192
0.448192 0.551808 0.448192
0.448192 0.655425 0.344575
0.344575 0.551808 0.551808
0.344575 0.655425 0.655425
0.448192 0.551808 0.655425
0.448192 0.655425 0.551808
0.551808 0.344575 0.344575
0.551808 0.448192 0.448192
0.655425 0.344575 0.448192
0.655425 0.448192 0.344575
0.551808 0.344575 0.551808
0.551808 0.448192 0.655425
0.655425 0.344575 0.655425
0.655425 0.448192 0.551808
0.551808 0.551808 0.344575
0.551808 0.655425 0.448192
0.655425 0.551808 0.448192
0.655425 0.655425 0.344575
0.551808 0.551808 0.551808
0.551808 0.655425 0.655425
0.655425 0.551808 0.655425
0.655425 0.655425 0.551808
0.448192 0.344575 0.344575
0.448192 0.448192 0.448192
0.344575 0.344575 0.448192
0.344575 0.448192 0.344575
0.448192 0.344575 0.551808
0.448192 0.448192 0.655425
0.344575 0.344575 0.655425
0.344575 0.448192 0.551808
0.448192 0.551808 0.344575
0.448192 0.655425 0.448192
0.344575 0.551808 0.448192
0.344575 0.655425 0.344575
0.448192 0.551808 0.551808
0.448192 0.655425 0.655425
0.344575 0.551808 0.655425
0.344575 0.655425 0.551808
0.655425 0.344575 0.344575
0.655425 0.448192 0.448192
0.551808 0.344575 0.448192
0.551808 0.448192 0.344575
0.655425 0.344575 0.551808
0.655425 0.448192 0.655425
0.551808 0.344575 0.655425
0.551808 0.448192 0.551808
0.655425 0.551808 0.344575
0.655425 0.655425 0.448192
0.551808 0.551808 0.448192
0.551808 0.655425 0.344575
0.655425 0.551808 0.551808
0.655425 0.655425 0.655425
0.551808 0.551808 0.655425
0.551808 0.655425 0.551808
'''
)
# with open("/home/xzhang1/m_cpu_config.log", "w") as of_:
#     of_.write('%15s %15s %15s %15s %15s %15s %15s %15s %15s %15s\n' %('proj_real','proj_reci', 'npar', 'wavef', 'kpar', 'arraygrid', 'ncore_total', 'estimate_mem', 'real_mem', 'ele_time'))

for ncore_node in [18, 16, 12]:
    for ncore in set(factor1(ncore_node)):
        for kpar in set(factor1(ncore)) & set(factor1(2)):
            for prec in ['Accurate','Normal','Low']:
                for lreal in ['False','True']:

                    n.property = '''
                    eigenfunction, engine=vasp, spin=para,
                    ediff=1E-3, isym=2, ismear=0, sigma=0.02,
                    ncore=%s, ncore_node=%s, prec=%s, encut=400, lreal=%s, kpoints=G 2 1 1, kpar=%s,
                    read=n, platform=dellpc
                    ''' %(ncore, ncore_node, prec, lreal, kpar)

                    n.reset()
                    n.compute()

                    with open("/home/xzhang1/m_cpu_config.log", "r") as if_:
                        attempt_line = '%15s %15s %15s %15s %15s %15s %15s %15s ' %(n.gen.memory['projector_real'], n.gen.memory['projector_reciprocal'], n.gen.getkw('npar'), n.gen.memory['wavefunction'], n.gen.getkw('kpar'), n.gen.memory['arraygrid'], int(n.gen.getkw('nnode')) * int(n.gen.getkw('ncore_node')) ,
                            ((n.gen.memory['projector_real'] + n.gen.memory['projector_reciprocal']) * float(n.gen.getkw('npar')) + n.gen.memory['wavefunction'] * float(n.gen.getkw('kpar'))) /1024.0/1024 + float(n.gen.getkw('nnode'))*700)
                        if any( [attempt_line in line for line in if_.readlines()] ):
                            continue

                    with open("/home/xzhang1/m_cpu_config.log", "a") as of_:
                        of_.write('%15s %15s %15s %15s %15s %15s %15s %15s ' %(n.gen.memory['projector_real'], n.gen.memory['projector_reciprocal'], n.gen.getkw('npar'), n.gen.memory['wavefunction'], n.gen.getkw('kpar'), n.gen.memory['arraygrid'], int(n.gen.getkw('nnode')) * int(n.gen.getkw('ncore_node')) ,
                        ((n.gen.memory['projector_real'] + n.gen.memory['projector_reciprocal']) * float(n.gen.getkw('npar')) + n.gen.memory['wavefunction'] * float(n.gen.getkw('kpar'))) /1024.0/1024 + float(n.gen.getkw('nnode'))*700) )
                    os.system('pkill vasp_std')
                    basic_mem = psutil.virtual_memory().used / 1000000.0
                    basic_time = time.time()
                    os.chdir(n.path)
                    os.system('./wrapper')
                    while True:
                        time.sleep(1)
                        with open("run.log","r") as if_:
                            if if_.readlines()[-1].startswith("DAV:"):
                                mem = psutil.virtual_memory().used / 1000000.0 - basic_mem
                                time1 = time.time()
                                with open("/home/xzhang1/m_cpu_config.log", "a") as of_:
                                    of_.write('%15s %15s\n' %(mem, time1 - basic_time))
                                os.system('pkill vasp_std')
                                break
