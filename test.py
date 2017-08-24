import dill as pickle   # dill requires citation
import re
import os
import time

import engine
import shared
import qchem

import os
import sys
import subprocess
import re
import numpy as np
import scipy
import shutil
from pprint import pprint
import tempfile
import hashlib
from subprocess import call, check_output, STDOUT, CalledProcessError
from filecmp import dircmp
from collections import OrderedDict
import paramiko

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import progressbar
from progressbar import Bar, Counter, ETA,FormatLabel, Percentage,ProgressBar 
from scipy.interpolate import Rbf
from scipy.optimize import minimize
from scipy import spatial

import shared


#with open('data/markdown') as f:
#    qchem.Import(f.read())

#with open('POSCAR','r') as f:
#    engine.Cell(f.read())

#qchem.Load()

#print shared.NODES['master'].map

#bqb = engine.Map().lookup('master.PbS QD.bare qd benchmarks')
#delattr(bqb, 'phase')

#qchem.Dump()

#ssh = paramiko.SSHClient()
#ssh.load_system_host_keys()
#ssh.connect('nanaimo', username='xzhang1')
#ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("squeue -n ssr")
#squeue_result = ssh_stdout.read().strip()
#vasp_is_running = ( len(squeue_result.splitlines()) > 1 )
