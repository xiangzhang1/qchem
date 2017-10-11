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

cell_text_list = ['''Pb S
1
26.0142	0.0	0.0
0.0	26.0142	0.0
0.0	0.0	26.0142
Pb  S
13  14
Direct
0.384406 0.384406 0.5
0.5 0.384406 0.384406
0.5 0.384406 0.615594
0.384406 0.5 0.384406
0.384406 0.615594 0.5
0.5 0.5 0.5
0.5 0.615594 0.384406
0.384406 0.5 0.615594
0.5 0.615594 0.615594
0.615594 0.384406 0.5
0.615594 0.5 0.384406
0.615594 0.615594 0.5
0.615594 0.5 0.615594
0.5 0.384406 0.5
0.384406 0.384406 0.384406
0.384406 0.384406 0.615594
0.5 0.5 0.384406
0.5 0.615594 0.5
0.384406 0.5 0.5
0.384406 0.615594 0.384406
0.5 0.5 0.615594
0.384406 0.615594 0.615594
0.615594 0.384406 0.384406
0.615594 0.384406 0.615594
0.615594 0.5 0.5
0.615594 0.615594 0.384406
0.615594 0.615594 0.615594
''',
'''Pb S
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
                    time.sleep(2)
                    with open("run.log","r") as if_:
                        text = if_.read()
                        lines = text.splitlines()
                        if lines[-1].startswith("DAV:") or lines[-1].startswith("RMM:"):
                            mem = float(re.findall('\d+', os.popen("nvidia-smi | sed -n '9p' | awk '{print $9}'").read())[0])
                            time1 = time.time()
                            with open("/home/xzhang1/m_gpu_config.log", "a") as of_:
                                of_.write('%15s %15s\n' %(mem, time1 - basic_time))
                            os.system('pkill vasp_gpu')
                            break
                        if 'segmentation fault occurred' in  text:
                            with open("/home/xzhang1/m_gpu_config.log", "a") as of_:
                                of_.write('segfault segfault\n' )
                                break
