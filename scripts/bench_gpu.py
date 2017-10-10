# import engine
import numpy as np
import os
import time
import subprocess
import re

def factor1(n):
    """returns a list of prime factors of n"""
    factors = []
    for i in range(1,n+1):
        if n % i == 0:
            factors.append(i)
    return factors

cell_text_list = ['''Pb4 S4
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
''',
'''Pb4 S8
1.0
6.175947 0.000000 0.000000
0.000000 6.175947 0.000000
0.000000 0.000000 7.637695
Pb S
4 8
Direct
0.000000 0.000000 0.750000
0.000000 0.000000 0.250000
0.500000 0.500000 0.250000
0.500000 0.500000 0.750000
0.619362 0.880638 0.500000
0.880638 0.619362 0.000000
0.880638 0.380638 0.500000
0.619362 0.119362 0.000000
0.119362 0.380638 0.000000
0.380638 0.119362 0.500000
0.380638 0.880638 0.000000
0.119362 0.619362 0.500000
''',
'''Pb2 S2
1.0
4.169239 0.000000 0.000000
-2.084619 3.610667 0.000000
0.000000 0.000000 7.341310
Pb S
2 2
Direct
0.000000 0.000000 0.500000
0.000000 0.000000 0.000000
0.333333 0.666667 0.250000
0.666667 0.333333 0.750000
''']





n = engine.Map().lookup('master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.static')

# with open("/home/xzhang1/m_gpu_config.log", "w") as of_:
#     of_.write('%15s %15s %15s %15s %15s %15s %15s %15s\n' %('proj_real','proj_reci', 'wavef', 'kpar', 'arraygrid', 'cpu_est_mem', 'real_mem', 'ele_time'))

for cell_text in cell_text_list:
    for prec in ['Accurate','Normal','Low']:
        for kpar in [1, 2, 4, 8]:
            for encut in [200, 300, 400]:
                n.cell = engine.Cell(cell_text)
                n.property = '''
                eigenfunction, engine=vasp, spin=para,
                ediff=1E-3, isym=2, ismear=-5,
                ncore=1, ncore_node=1, prec=%s, encut=%s, lreal=On, kpar=%s, kpoints=G 2 2 2, algo=Fast, 
                read=n, platform=dellpc, gpu
                ''' %(prec, encut, kpar)
                n.phase = 'pbs, bulk'

                n.reset()
                n.compute()

                with open("/home/xzhang1/m_gpu_config.log", "r") as if_:
                    attempt_line = '%15s %15s %15s %15s %15s %15s ' %(n.gen.memory['projector_real'], n.gen.memory['projector_reciprocal'], n.gen.memory['wavefunction'], n.gen.getkw('kpar'), n.gen.memory['arraygrid'],
                        ((n.gen.memory['projector_real'] + n.gen.memory['projector_reciprocal']) * float(n.gen.getkw('npar')) + n.gen.memory['wavefunction'] * float(n.gen.getkw('kpar'))) /1024.0/1024 + float(n.gen.getkw('nnode'))*700)
                    if any( [attempt_line in line for line in if_.readlines()] ):
                        continue

                with open("/home/xzhang1/m_gpu_config.log", "a") as of_:
                    of_.write('%15s %15s %15s %15s %15s %15s ' %(n.gen.memory['projector_real'], n.gen.memory['projector_reciprocal'], n.gen.memory['wavefunction'], n.gen.getkw('kpar'), n.gen.memory['arraygrid'],
                        ((n.gen.memory['projector_real'] + n.gen.memory['projector_reciprocal']) * float(n.gen.getkw('npar')) + n.gen.memory['wavefunction'] * float(n.gen.getkw('kpar'))) /1024.0/1024 + float(n.gen.getkw('nnode'))*700) )
                os.system('pkill vasp_gpu')
                basic_mem = float(re.findall('\d+', os.popen("nvidia-smi | sed -n '9p' | awk '{print $9}'").read())[0])
                basic_time = time.time()
                os.chdir(n.path)
                os.system('./wrapper')
                while True:
                    time.sleep(1)
                    with open("run.log","r") as if_:
                        if if_.readlines()[-1].startswith("DAV:"):
                            mem = float(re.findall('\d+', os.popen("nvidia-smi | sed -n '9p' | awk '{print $9}'").read())[0])
                            time1 = time.time()
                            with open("/home/xzhang1/m_gpu_config.log", "a") as of_:
                                of_.write('%15s %15s\n' %(mem, time1 - basic_time))
                            os.system('pkill vasp_gpu')
                            break
