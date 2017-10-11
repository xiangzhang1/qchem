# import engine
import numpy as np
import os
import time
import psutil

import itertools

def factor1(n):
    """returns a list of prime factors of n"""
    factors = []
    for i in range(1,n+1):
        if n % i == 0:
            factors.append(i)
    return factors

def GenCh100(nx, ny, nz, padding, a=3.007083725):
    # base
    cellar = np.diag([a, a, a])
    base = np.diag([nx-1, ny-1, nz-1]) * cellar
    # padding
    base += np.diag([padding, padding, padding])
    # ccoor, atoms
    ccoor = []
    atoms = []
    for ix, iy, iz in np.ndindex(nx, ny, nz):
        ccoor_ = np.dot(np.float_([ix, iy, iz]), cellar)     # numpy auto-collapses 1d array.
        ccoor.append(ccoor_.tolist())
        atoms.append('Pb' if (ix+iy+iz)%2 else 'S')
    ccoor = np.float_(ccoor)
    # padding
    shift = np.dot(np.float_([0.5, 0.5, 0.5]), base) - np.mean(ccoor, axis=0)
    ccoor[:] += shift
    # fcoor
    fcoor = np.dot(ccoor, np.linalg.inv(base))
    # stoichiometry
    stoichiometry = dict([(key, len(list(group))) for key, group in itertools.groupby(sorted(atoms))])
    # Cell() text
    text = 'PbS QD\n'
    text += '1\n'
    for v in base:
        text += ' '.join(map(str, v)) + '\n'
    text += ' '.join(stoichiometry.keys()) + '\n'
    text += ' '.join(map(str, stoichiometry.values())) + '\n'
    text += 'Direct\n'
    for symbol in stoichiometry.keys():
        for idx_atoms, atoms_ in enumerate(atoms):
            if atoms_ == symbol:
                text += ' '.join(map(str, fcoor[idx_atoms])) + '\n'
    return text


n = engine.Map().lookup('master.PbS QD.bare qd testing.Q0 Test convergence.Pb55S38.static0')

for ncore_node in [18, 16, 12]:
    for ncore in set(factor1(ncore_node)):
        for prec in ['Normal','Low']:
            with open("/home/xzhang1/oa.txt","r") as if_:
                for line in if_.readlines()[1:]:
                    # nx, ny, nz, padding
                    nx, ny, nz, padding = map(float, line.split())

                    n.property = '''
                    eigenfunction, engine=vasp, spin=para,
                    ediff=1E-3, isym=2, ismear=0, sigma=0.02,
                    ncore=%s, ncore_node=%s, prec=%s, encut=200, lreal=False, kpoints=G 1 1 1, kpar=1,
                    read=n, platform=dellpc
                    ''' %(ncore, ncore_node, prec, lreal, kpar)
                    n.phase = 'pbs, qd'
                    n.cell = engine.Cell(GenCh100(nx, ny, nz, padding))

                    n.reset()
                    n.compute()

                    # write part 1
                    with open("/home/xzhang1/m_cpu_config.log", "r") as if_:
                        attempt_line = '%15s %15s %15s %15s %15s %15s %15s %15s %15s ' %(
                            n.gen.memory['projector_real'],
                            n.gen.memory['projector_reciprocal'],
                            n.gen.memory['wavefunction'],
                            n.gen.memory['arraygrid'],
                            n.cell.natoms,
                            np.dot(np.cross(cell.base[0], cell.base[1]), cell.base[2]),
                            n.gen.getkw('npar'),
                            n.gen.getkw('ncore_node') ,
                            ((n.gen.memory['projector_real'] + n.gen.memory['projector_reciprocal']) * float(n.gen.getkw('npar')) + n.gen.memory['wavefunction'] * float(n.gen.getkw('kpar'))) /1024.0/1024 + 700)
                        if any( [attempt_line in line for line in if_.readlines()] ):
                            continue
                        else:
                            of_.write(attempt_line)

                    # write part 2
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
