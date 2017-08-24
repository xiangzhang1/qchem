#!/usr/bin/python
'''CLI, as a backup option during downtime'''

# flask
from flask import Flask, request, jsonify, session, Response
from flask_cors import CORS, cross_origin
from functools import wraps

# flask over https
from OpenSSL import SSL
context = SSL.Context(SSL.SSLv23_METHOD)
context.use_privatekey_file('yourserver.key')
context.use_certificate_file('yourserver.crt')

# common libraries
import sys
import os
import shutil
import random
import string
import dill as pickle
import time
from pprint import pprint
import IPython

from cStringIO import StringIO
from fuzzywuzzy import process

# qchem package
import qchem
import shared
import engine
from shared import ELEMENTS

IPython.embed()

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
