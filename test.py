import shared
import sys
import paramiko
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

ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
ssh.connect('nanaimo', username='xzhang1')
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("squeue -n ssr")
squeue_result = ssh_stdout.read().strip()
vasp_is_running = ( len(squeue_result.splitlines()) > 1 )
