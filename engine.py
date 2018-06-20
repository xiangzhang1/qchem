#!/usr/bin/env python
# -*- coding: utf-8 -*-

# scientific kit
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
import pandas as pd
import matplotlib.pyplot as plt

# date
import time
import datetime

# sys
import os
import sys
import subprocess
import shutil

# generic
import re
from pprint import pprint
import tempfile
import hashlib
from subprocess import call, check_output, STDOUT, CalledProcessError
from filecmp import dircmp
from collections import OrderedDict
import paramiko
import IPython
from itertools import groupby, chain, combinations
from tqdm import tqdm, trange
import math
import string
import random
from retrying import retry
import requests
import copy
from functools import partial
from weakref import ref
from textwrap import dedent
import itertools

# matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

# scipy
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy import spatial
from scipy.linalg import norm

# qchem
import shared

# ase: some of its libraries are fantastic.
import ase

'''
TOC
===========================================================================
Gen, Makeparam
Cell
Map
Dummy
Vasp, ssh_and_run
Electron - Grepen, Dos, Bands, Charge, Errors
compare_and_X, Compare
'''

# Gen
# ===========================================================================


class Gen(object):  # Stores the logical structure of keywords and modules. A unique construct deserving a name.

    # 1. getkw, parse_if, write_incar_kpoints
    # ---------------------------------------
    def getkw(self, kwname):
        if kwname not in self.kw:
            raise shared.DeferError(self.__class__.__name__ + '. getkw: keyword {%s} not found, DeferError raised' %(kwname))
        if len(self.kw[kwname]) != 1:
            raise shared.DeferError(self.__class__.__name__ + '.getkw: self.kw[%s] does not have 1 unique value {%s}, DeferError raised' %(kwname, self.kw[kwname]))
        if not isinstance(next(iter(self.kw[kwname])), basestring):
            raise shared.CustomError(self.__class__.__name__ + '.getkw: value {%s} of kw {%s} is not string' % (next(iter(self.kw[kwname])), kwname))
        return next(iter(self.kw[kwname]))

    def evaluate(self, expression):
        # Evaluates expression to string: literal or (funcname)
        if expression.startswith('(') and expression.endswith(')'):
            func = getattr(self, re.sub('\(|\)', '', expression).strip())
            if not func:
                raise shared.CustomError(self.__class__.__name__+' error:  bad conf file error. Unable to find function {%s}' % re.sub('\(|\)', '', expression))
            return func()
        else:   # literal
            return expression

    def parse_require(self, expression, run=False):  # Executes single require expression. Accepts empty expression as True.
        if len(expression.splitlines()) > 1:    # grammar check
            raise shared.CustomError(self.__class__.__name__ + '.parse_require: expression {%s} contains line break' %expression)
        operation_map = {
                '=': lambda x, y: x & y,
                '!=': lambda x, y: x - y,
                }
        expression = expression.strip()
        if re.search('=', expression):           # evaluation        ## parse kwname!=kwval|(funcname)
            l = [ p.strip() for p in re.split('(!=|=)', expression)]; kwname = l[0] ; operator = l[1] ; kwval_expression = l[2]
            kwvalset = set()
            for kwval_part in kwval_expression.split('|'):
                kwval_part = kwval_part.strip()
                kwvalset.add(self.evaluate(kwval_part))
            if kwname in self.kw and bool(self.kw[kwname]):
                result = operation_map[operator](self.kw[kwname], kwvalset)
            else:
                result = operation_map[operator](kwvalset, kwvalset)
            if run and bool(result):
                self.kw[kwname] = result
                if shared.DEBUG >= 1:
                    print self.__class__.__name__ + ' parse_require: gave kw {%s} value {%s}' % (kwname, result)
            if run and not bool(result):
                raise shared.CustomError(self.__class__.__name__ + ' parse_require run=True error: parse_require results in empty set: kwname {%s}, value {%s}, required value {%s}' % (kwname, self.kw[kwname] if kwname in self.kw else 'null', kwvalset))
            if not run and not bool(result) and shared.DEBUG >= 1:
                print self.__class__.__name__ + ' parse_require warning: parse_require results in empty set, deferred: kwname {%s}, value {%s}, required_value {%s}' %(kwname, self.kw[kwname] if kwname in self.kw else 'null', kwvalset)
            self.kw_legal_set.add(kwname)
            return bool(result)
        elif 'internal' in expression:      ## parse kwname internal
            kwname = re.sub('internal', '', expression).strip()
            self.kw_internal_set.add(kwname)
            return True
        elif not '(' in expression and not 'null' in expression:    ## parse !modname
            modname = re.sub('!', '', expression).strip()
            if '!' in expression:
                result = (self.mod[modname] if modname in self.mod else set()) - set([True])
            else:
                result = (self.mod[modname] if modname in self.mod else set()) | set([True])
            if run and bool(result):        ### output
                self.mod[modname] = result
            self.mod_legal_set.add(modname)
            return bool(result)
        else:                               ## parse if expression
            result = self.parse_if(expression)
            if not run and not result and shared.DEBUG >= 1:
                    print self.__class__.__name__ + ' parse_require warning: parse_require results in empty set, deferred: expression {%s}' %(expression)
            return result

    def parse_if(self,expression):  # recursively evaluate complex if condition. accepts empty expression.
        if ',' in expression:
            raise shared.CustomError( self.__class__.__name__ + ' parse_if error: "," in if expression {%s} in engine.gen.*.conf. Did you mean to use "&"?' %expression)
        operation_map = {
                '&&': lambda x, y: x and y,
                '||': lambda x, y: x or y,
                '&': lambda x, y: x and y,
                '|': lambda x, y: x or y
                }
        expression = expression.strip()
        if re.search('&&|\|\|', expression):
            l = [ x.strip() for x in re.split('(&&|\|\|)', expression,maxsplit=1) ]
            return operation_map[l[1]](self.parse_if(l[0]), self.parse_if(l[2]))
        elif re.search('&|\|', expression):
            l = [ x.strip() for x in re.split('(&|\|)', expression, maxsplit=1) ]
            return operation_map[l[1]](self.parse_if(l[0]), self.parse_if(l[2]))
        elif '!' in expression:
            return not self.parse_if(expression.replace('!', ''))
        elif '=' in expression:  # evaluation    ## parse kwname=kwval
            name = expression.split('=')[0].strip()
            val = expression.split('=')[1].strip()
            result = (name in self.kw and self.kw[name]==set([val]))
            self.kw_legal_set.add(name)
            return result
        elif expression.startswith('(') and expression.endswith(')'):    ## parse (funcname)
            return self.evaluate(expression)
        elif expression == '':                  ## parse empty expression
            return True
        elif 'null' in expression:              ## parse kwname null
            kwname = re.sub('null', '', expression).strip()
            result = not (kwname in self.kw and bool(self.kw[kwname]) )
            self.kw_legal_set.add(kwname)
            return result
        else:                                   ## parse modname
            self.mod_legal_set.add(expression)
            if expression in self.mod and self.mod[expression]==set([True]):
                return True
            else:  #not defined means not written, which means no
                return False

    def write_incar_kpoints(self):
        with open('INCAR','w') as outfile:
            for name in self.kw:
                if name not in self.kw_internal_set:
                    outfile.write('    '+name.upper()+' = '+str(self.getkw(name))+'\n')
        with open('KPOINTS','w') as outfile:
            kpoints = self.getkw('kpoints').split()
            if kpoints[0] in 'GM':
                outfile.write('KPOINTS\n')
                outfile.write('0\n')
                outfile.write(kpoints[0] + '\n')
                outfile.write( ' '.join(kpoints[1:]) + '\n' )
                outfile.write('0 0 0')
            elif kpoints[0] in 'L':
                outfile.write('KPOINTS\n%s\nline\nreciprocal\n' %kpoints[1])
                high_symmetry_symbol_dict = {
                    'B1' : {
                        'G': '0 0 0',
                        'X': '0 0 0.5',
                        'W': '0 0.5 0.5',
                        'L': '0.5 0.5 0.5'
                    }
                }
                for idx in range(2, len(kpoints)-1):
                    outfile.write(high_symmetry_symbol_dict[self.getkw('struk')][kpoints[idx]]+'\n')
                    outfile.write(high_symmetry_symbol_dict[self.getkw('struk')][kpoints[idx+1]]+'\n')
                    outfile.write('\n')
            else:
                raise shared.CustomError(self.__class__.__name__ + '.write_incar_kpoints: kpoints starter looks wrong')

    def pot(self, symbol):
        if len(shared.ELEMENTS[symbol].pot) == 0:
            raise shared.CustomError(' pot: POTCAR for '+symbol+' not found.')
        path = shared.script_dir + '/resource/potpaw_PBE/'+shared.ELEMENTS[symbol].pot + '/POTCAR'
        if_ = open(path,'r')
        of_ = open('./POTCAR','a')
        of_.write( if_.read() )

    def __str__(self):
        result = ''
        for name in self.mod:
            if self.parse_if(name):
                result += name + ', '
        for name in self.kw:
            result += name + '=' + self.getkw(name) + ', '
        return result


    # 2. construct main data structure ; perform add-on analysis
    # ----------------------------------------------------------

    def __init__(self, node):
        self.node = ref(node)
        cell = node.cell
        input_ = node.phase + ', ' + node.property
    # 读mod, kw
        self.mod = {}
        self.kw = {}
        self.kw_legal_set = set()
        self.kw_internal_set = set()
        self.mod_legal_set = set()
        input_ = [p.strip() for p in input_.split(',') if p.rstrip()]
        for item in input_:
            self.parse_require(item,True)
    # 执行require
        self.require = []
        if not [x for x in input_ if x.startswith('engine')]:
            raise shared.CustomError(self.__class__.__name__+': __init__: no engine=x found. Input_: {%s}' %input_)
        engine_name = [x for x in input_ if x.startswith('engine')][0].split('=')[1].strip()
        with open(shared.script_dir + '/conf/engine.gen.' + engine_name + '.conf') as conf:
            lines = conf.read().splitlines()
            for line in [ [p.strip() for p in l.split(':')] for l in lines if not l.startswith('#') ]:
                if len(line) < 4: raise shared.CustomError('bad conf grammar error: needs 3 colons per line least in {%s}' %line)
                for part in [p.strip() for p in line[1].split(',') ]:
                    try:
                        if self.parse_if(line[0]) and self.parse_require(part, run=False) and line[2 ]!='optional':
                            self.parse_require(part, run=True)
                        else:
                            self.require.append([line[0],part,line[2],line[3]])
                    except shared.DeferError:
                            self.require.append([line[0],part,line[2],line[3]])
        ## round 2+: got a 'no' or 'optional' in first round
        continue_flag = True
        while continue_flag:
            continue_flag = False
            for line in self.require:
                try:
                    if self.parse_if(line[0]) and self.parse_require(line[1],False):
                        self.parse_require(line[1],True)
                        continue_flag = True
                        self.require.remove(line)
                except shared.DeferError:
                    pass
        ## round last: the 'no' is final
        for line in self.require:
            if self.parse_if(line[0]):
                if line[2] == 'optional':
                    print self.__class__.__name__+' __init__ round 2 warning: parse_require result in empty set. optional and aborted. Expression is { %s : %s : %s }.' % (line[0],line[1],line[3])
                else:
                    raise shared.CustomError( self.__class__.__name__+' __init__ round 2 error: parse_require still produces empty set. Expression is { %s : %s :  %s }.' % (line[0],line[1],line[3]) )
        #  检验
        ## Entering the last phase. Data structure of self.mod and self.kw simplifies.
        for modname in set(self.mod.keys())-self.mod_legal_set:
            print self.__class__.__name__+' warning: illegal name. Mod {%s} is not in mod_legal_set and has been ignored. Mod_legal_set={%s}' %(modname,self.mod_legal_set)
            del self.mod[modname]
        for kwname in set(self.kw.keys())-self.kw_legal_set:
            print self.__class__.__name__+' warning: illegal name. Kw {%s} is not required and has been ignored.' % kwname
            del self.kw[kwname]
        for name in self.kw:
            if len(self.kw[name]) != 1:
                raise shared.CustomError( self.__class__.__name__+' error: non-unique output. Kw[%s]={%s} has not been restricted to 1 value.' %(name,self.kw[name]) )
        # Overhead prediction. Unfortunately, what should have been done in vasp is moved here, for high efficiency.
        if self.parse_if('engine=vasp'):
            print '-' * 30 + ' Overheads ' + '-' * 30
            # Memory
            m = Makeparam(self)
            memory_predicted_gb = ( (m.projector_real + m.projector_reciprocal)*int(self.getkw('npar')) + m.wavefunction*float(self.getkw('kpar')) )/1024.0/1024/1024 + int(self.getkw('nnode'))*0.7    # Global data cannot be obtained for multi-node multi-CPU case. ML is not suitable. Memory estimations might just also be applicable to GPU.
            memory_available_gb = int(self.getkw('nnode')) * int(self.getkw('mem_node'))
            print self.__class__.__name__ + ' memory %s: %s GB used out of %s GB' %('prediction' if memory_available_gb>memory_predicted_gb else 'WARNING', memory_predicted_gb, memory_available_gb)
            # Queue time; run time

    # 3. nbands, ncore_total, encut
    # -----------------------------
    def ncore_total(self):
        return str( int(self.getkw('nnode')) * int(self.getkw('ncore_node')) )

    def nbands(self):
        cell = self.node().cell
        print self.__class__.__name__ + ' warning: nbands may not be that reliable'
        # extracted from vasp source code
        if self.parse_if('spin=ncl'):
            nbands = ( cell.nelectrons() * 3 / 5 + sum(cell.stoichiometry.values()) * 3 / 2 ) * 2
        elif self.parse_if('spin=para'):
            nbands = cell.nelectrons() * 3 / 5 + sum(cell.stoichiometry.values()) * 1 / 2
        elif self.parse_if('spin=afm|spin=fm'):
            nbands = cell.nelectrons() / 2 + sum(cell.stoichiometry.values()) / 2
        else:
            raise shared.CustomError(self.__class__.__name__+'spin variable is not fm, afm or para, cannot compute nbands')
        # nbands change based on parallel
        nbands = math.ceil(nbands / int(self.getkw('npar'))) * int(self.getkw('npar'))
        return str(int(nbands))

    def lmaxmix(self):
        cell = self.node().cell
        b_l_map = { 's': 2, 'p': 2, 'd': 4, 'f': 6, 'g': 8 }
        lmaxmix = max( [ b_l_map[ shared.ELEMENTS[symbol].block ] for symbol in  cell.stoichiometry.keys() ] )
        return str(lmaxmix)

    def encut(self):
        cell = self.node().cell
        result = max( [ shared.ELEMENTS[symbol].pot_encut for symbol in cell.stoichiometry.keys() ] )
        if result + 140 < 400:  # totally confusing guess, out of perovskites and quantum dots
            result = 400
        elif result < 400:
            result += 140
        else:
            result += 50
        return str(result)

    def ismear5check(self):
        '''kpoints is fit for ismear=5'''
        kpoints = self.getkw('kpoints').split(' ')
        if kpoints[0] in 'GM':
            return np.prod([int(x) for x in kpoints[1:] ]) > 2
        else:
            return False

    def kpointscheck(self):
        '''kpoints format is sane'''
        kpoints = self.getkw('kpoints').split()
        if kpoints[0] in 'GM' and len(kpoints)==4:
            return True
        elif kpoints[0] in 'L' and len(kpoints)>2:
            return True
        else:
            raise shared.CustomError(self.__class__.__name__ + '.kpointscheck: kpoints format wrong. ')
        if kpoints[0] == 'M':
            print self.__class__.__name__ + '.kpointscheck warning: In general, for low-symmetry cells it is sometimes difficult to symmetrize the k-mesh if it is not centered on Gamma. For hexagonal cell, it becomes indeed impossible.'
        return True

    def qdoptwarning(self):
        print self.__class__.__name__ + '.qdoptwarning: 认为存在一个半稳定点，和轻微平凡异构，需要小心。'
        return True

    def nkred_divide(self):
        kpoints = self.getkw('kpoints').split(' ')
        return kpoints[0] in 'GM'\
               and int(kpoints[1]) % int(self.getkw('nkredx')) == 0 \
               and int(kpoints[2]) % int(self.getkw('nkredy')) == 0 \
               and int(kpoints[3]) % int(self.getkw('nkredz')) == 0

    def magmom(self):
        cell = self.node().cell
        magmom = ''
        if self.parse_if('spin=afm'):
            print self.__class__.__name__ + ' warning: more than 1 AFM pattern exists.'
            for symbol in cell.stoichiometry:
                l = [0] * cell.stoichiometry[symbol]
                base = shared.ELEMENTS[symbol].magmom
                l[::2] = base
                l[1::2] = -1 * base
                magmom += ' ' + ' '.join(l)
        if self.parse_if('spin=fm'):
            for symbol in cell.stoichiometry:
                base = shared.ELEMENTS[symbol].magmom
                magmom += ' ' + str( cell.stoichiometry[symbol] ) + '*' + str( base )
        return magmom
    def ldauu(self):
        ldauu = ''
        for symbol in self.node().cell.stoichiometry:
            ldauu += str( shared.ELEMENTS[symbol].ldauu )
        return ldauu
    def ldauj(self):
        ldauj = ''
        for symbol in self.node().cell.stoichiometry:
            ldauj += str( shared.ELEMENTS[symbol].ldauj )
        return ldauj

    # 5. archived functions - may be called externally
    def npar(self):
        return str( int(self.getkw('ncore_node')) * int(self.getkw('nnode')) / int(self.getkw('ncore')) )
    def ncore(self):
        return str( int(self.getkw('ncore_node')) * int(self.getkw('nnode')) / int(self.getkw('npar')) )

# Makeparam: check memory based on gen.
# -------------------------------------
class Makeparam(object):

    def __init__(self, gen):
        # pre-conditioning
        tmp_gen = copy.deepcopy(gen)
        tmp_gen.kw['lsorbit'] = ['.FALSE.']
        tmp_gen.kw['isym'] = ['0']
        # make dir, write files
        tmp_path = shared.script_dir + '/check_memory_tmp' + ''.join(random.sample(string.ascii_lowercase,4))
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        os.mkdir(tmp_path)
        os.chdir(tmp_path)
        tmp_gen.write_incar_kpoints()
        with open('POSCAR','w') as f:
            f.write(gen.node().cell.poscar4())
        for symbol in gen.node().cell.stoichiometry.keys():
            gen.pot(symbol)
        # parse output
        try:
            output = check_output([shared.script_dir + '/resource/makeparam']).splitlines()
        except CalledProcessError:
            raise shared.CustomError(self.__class__.__name__ + ' error: makeparam failed with non-zero exit status. possibly memory error. give it up man :)')
        try:
            self.arraygrid = int( next(l for l in output if 'arrays on large grid' in l).split()[7] )
            self.wavefunction = int( next(l for l in output if 'sets of wavefunctions' in l).split()[4] )
            self.projector_real = abs(int( next(l for l in output if 'projectors in real space' in l).split()[4] ))
            self.projector_reciprocal = abs(int( next(l for l in output if 'projectors in reciprocal space' in l).split()[4] ))
        except StopIteration, KeyError:
            print '\n'.join(output)
            raise shared.CustomError(tmp_gen.__class__.__name__ + ' error: makeparam output illegal. Check POSCAR4 format and memory leak in directory {%s}.' %tmp_path)
        # cleanup
        os.chdir(shared.script_dir)
        shutil.rmtree(tmp_path)




# ===========================================================================


# Cell: stores and parses atomic configuration file.

class Cell(object):

    def __init__(self,lines):

        # basics
        if '\n' in lines:
            lines = lines.splitlines()
        self.name = lines[0]
        self.base = np.float32([ line.split() for line in lines[2:5] ]) * float(lines[1])
        self.stoichiometry = OrderedDict( zip(lines[5].split(), [int(x) for x in lines[6].split()]) )

        # fork parsing on Direct | Selective Dynamics
        if lines[7].startswith('Select') and all(all(sel=='T' for sel in line.split()[3:]) for line in lines[9:9+sum(self.stoichiometry.values())] ):
            print self.__class__.__name__ + '.__init__: Selective dynamics cell, all T. Converting to trivial cell...'
            lines.pop(7)
        if lines[7].startswith('D'):
            fcoor = np.float32([ line.split()[:3] for line in lines[8:8+sum(self.stoichiometry.values())] ])
            self.ccoor = np.dot(fcoor, self.base)
            for fcoor_ in fcoor:
                if len(fcoor_)!=3:
                    raise shared.CustomError(self.__class__.__name__+'__init__: bad format. Coordinate line {%s}' %fcoor_)
        else:
            raise shared.CustomError(self.__class__.__name__+'__init__: unsupported POSCAR5 format. ')

    def fcoor(self):
        return np.dot(self.ccoor, np.linalg.inv(self.base))

    @shared.MWT(timeout=2592000)
    def cdist(self):
        return spatial.distance.squareform(spatial.distance.pdist(self.ccoor))

    @shared.MWT(timeout=2592000)
    def ccoor_mindist(self):
        return np.amin( spatial.distance.pdist(self.ccoor) )

    @shared.MWT(timeout=2592000)
    def ccoor_kdtree(self):
        return spatial.cKDTree( self.ccoor )

    def natoms(self):
        return sum( self.stoichiometry.values() )

    def nelectrons(self):
        return sum( [self.stoichiometry[symbol] * shared.ELEMENTS[symbol].pot_zval for symbol in self.stoichiometry] )

    def __str__(self):

        result = self.name+'\n'
        result += '1\n'
        for line in self.base:
            result += '\t'.join(map(str,line)) + '\n'
        result += '  '.join(self.stoichiometry.keys()) + '\n'
        result += '  '.join(map(str,self.stoichiometry.values())) + '\n'
        result += 'Direct\n'
        for line in np.dot(self.ccoor, np.linalg.inv(self.base)):
            result += ' '.join(map(str,line))+'\n'
        return result

    def poscar4(self):
        result = str(self)
        result = '\n'.join( [x for i,x in enumerate(result.splitlines()) if i!=5] )
        return result

    def symb_sgn(self):
        return list(itertools.chain.from_iterable([[symbol] * atomcount for symbol,atomcount in self.stoichiometry.items()]))




# ===========================================================================


class Map(object):

    def rlookup(self, attr_dict={}, node_list=[], parent=False, prev=False, prev2=False):
        '''
        reverse lookup. find the node specified by attr_dict and is in node_list.
        if parent=True or prev=True or prev2=True, find their respective relatives.
        end result is expected to be unique.
        '''
        l = self.lookup('master').map.traverse()
        l.add(self.lookup('master'))
        # basic checks
        if not attr_dict and not node_list:
            raise shared.CustomError(self.__class__.__name__ + '.rlookup: empty attr_dict and node_list provided')
        # find the node specified by attr_dict OR node_list
        primary = set()
        for n in l:
            if node_list and n in node_list:
                primary.add(n)
            if attr_dict and all( [getattr(n,key,None)==attr_dict[key] for key in attr_dict] ):
                primary.add(n)
        result = primary
        # parent=True: find common parent
        if parent:
            parents = set()
            for n in l:
                if getattr(n,'map',None) and all([x in n.map for x in primary]):
                    parents.add(n)
            result = parents
        # prev=True: find prev
        elif prev:
            prevs = set()
            parent = Map().rlookup(node_list=list(primary), parent=True)
            for n in parent.map._dict:
                if parent.map._dict[n] and all([x in parent.map._dict[n] for x in primary]):
                    prevs.add(n)
            result = prevs
        # prev2=True: find prev with type-2 link
        elif prev2:
            prev2s = set()
            parent = Map().rlookup(node_list=list(primary), parent=True)
            for n in parent.map._dict2:
                if parent.map._dict2[n] and all([x in parent.map._dict2[n] for x in primary]):
                    prev2s.add(n)
            result = prev2s
        else:
            pass
        # post-process
        if len(result)>1:
            raise shared.CustomError('RLookup: result is not unique. Criterion is: attr_dict: %s, node_list: %s, parent=%s, prev=%s, prev2=%s' %(attr_dict, [x.name for x in node_list]), parent, prev, prev2)
        return next(iter(result)) if result else None

    def lookup(self, name):
        if name == 'master':
            if name in shared.nodes:   return shared.nodes['master']
            else: raise shared.CustomError('找不到master了，求喂食')
        elif name in shared.nodes:
            return shared.nodes.pop(name)
        elif any([x.name == name for x in self._dict]):
            return [x for x in self._dict if x.name == name][0]
        elif '.' in name:
            return self.lookup('.'.join(name.split('.')[:-1])).map.lookup(name.split('.')[-1])
        else:
            raise LookupError(self.__class__.__name__ + ' lookup: Node %s not found' %name)

    def traverse(self):
        result = set([x for x in self])
        for n in [x for x in self._dict if getattr(x,'map',None)]:
            result = result | set( n.map.traverse() )
        return result

    def __init__(self, text=''):
        # initialize dict and text
        self._dict, self._dict2 = {}, {}
        text = text.split('\n')

        # src -> dst
        for line in text:
            if not line.rstrip():   continue
            line = [x.strip() for x in re.split('(->|-->)', line)]
            # lone node case
            if len(line) == 1:
                src = self.lookup(line[0])
                if src not in self._dict:   self._dict[src] = []
            elif len(line) == 3:
                src, dst = self.lookup(line[0]), self.lookup(line[2])
                # add src, dst to dict
                if src not in self._dict:
                    self._dict[src] = []
                if dst not in self._dict:
                    self._dict[dst] = []
                m = self._dict if line[1]=='->' else self._dict2
                m[src] = [dst] if src not in m else m[src]+[dst]
            else:
                raise shared.CustomError(self.__class__.__name__ + '__init__: src -> dst. 3 parts needed')


    def add_node(self, node):
        # inherit is done on compute
        # same name / same node exceptions are not allowed.
        # we're moving references around, so renaming is bad. instead, use 'duplicate' command intead.
        if any([x.name==node.name for x in self._dict]):
            raise shared.CustomError(self.__class__.__name__+' add_node: node with name {%s} already in self._dict. We\'re moving references around, so auto-renaming is bad. Use duplicate if only input is needed.' %node.name)
        else:
            self._dict[node] = []

    def del_node(self, node):
        #if [n for n in self._dict if n.name==name]:
        #    node = [n for n in self._dict if n.name==name][0]
        #else:
        #    raise shared.CustomError(self.__class__.__name__ + ' del_node error: node {%s} not found in self. Since we are deleting by name, we had better be sure.' %name)
        for m in (self._dict, self._dict2):
            m.pop(node,None)
            for n in m:
                m[n] = [x for x in m[n] if x != node]

    def add_edge(self, src_name, dst_name):
        src = self.lookup(src_name)
        dst = self.lookup(dst_name)
        if src in self._dict[dst] or (dst in self._dict2 and src in self._dict2[dst]):
            raise shared.CustomError(self.__class__.__name__ + ' add_edge: dst %s -> src %s link exists' %(dst_name, src_name))
        if dst in self._dict[src]:
            self._dict[src].remove(dst)
            self._dict2[src] = self._dict2[src]+[dst] if src in self._dict2 else [dst]
        elif src in self._dict2 and dst in self._dict2[src]:
            self._dict2[src].remove(dst)
            self._dict[src] = self._dict[src]+[dst] if src in self._dict else [dst]
        else:
            self._dict[src] += [dst]

    def del_edge(self, src_name, dst_name):
        src = self.lookup(src_name)
        dst = self.lookup(dst_name)
        if src in self._dict[dst] or dst in self._dict2 and src in self._dict2[dst]:
            raise shared.CustomError(self.__class__.__name__ + ' del_edge: dst %s -> src %s link exists' %(dst_name, src_name))
        if dst in self._dict[src]:
            self._dict[src].remove(dst)
        elif dst in self._dict2[src]:
            self._dict2[src].remove(dst)
        else:
            raise shared.CustomError(self.__class__.__name__ + ' del_edge: no edge to delete')

    def __str__(self):
        result = ''
        for src in self._dict:
            if not self._dict[src]:
                result += src.name + '\n'
            for dst in self._dict[src]:
                result += src.name + '->' + dst.name + '\n'
        for src in self._dict2:
            for dst in self._dict2[src]:
                result += src.name + '-->' + dst.name + '\n'
        return result

    def __contains__(self, item):
        return item in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __getitem__(self, key):
        return self._dict[key]




# Vasp
# ===========================================================================

#@retry(wait_random_min=1000, wait_random_max=2000, stop_max_attempt_number=10)
def ssh_and_run(platform, command):
    if platform not in ['dellpc', 'nanaimo', 'irmik']:
        raise shared.CustomError('ssh_and_run: unsupported platform %s' %platform)
    # paramiko ssh run command
    if platform == 'dellpc':
        return os.popen(command).read()
    else:
        try:
            ssh = shared.sshs[platform]
            if not ssh.get_transport().is_active():
                raise shared.CustomError
        except (KeyError, shared.CustomError) as e:
            ssh = shared.sshs[platform] = paramiko.SSHClient()
            ssh._policy = paramiko.WarningPolicy()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_config = paramiko.SSHConfig()
            user_config_file = os.path.expanduser("~/.ssh/config")
            if os.path.exists(user_config_file):
                with open(user_config_file) as f:
                    ssh_config.parse(f)
            ssh.load_system_host_keys()
            ssh.connect(platform, username='xzhang1')
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
        return ssh_stdout.read()


class Vasp(object):

    def __init__(self, node):
        self.node = ref(node)

    def compute(self):
        if shared.DEBUG>=2:    print 'calling %s(%s).compute' %(self.__class__.__name__, getattr(self,'path',''))

        node = self.node()
        prev = Map().rlookup(node_list=[node], prev=True)
        path = node.path
        gen = node.gen
        name = node.name

        if not getattr(self, 'wrapper', None):
            if os.path.exists(path):
                raise shared.CustomError( self.__class__.__name__ + ' __init__: path {%s} already exists. enforcing strictly, you need to remove it manually.' %path )
            os.makedirs(path)
            os.chdir(path)
            if gen.parse_if('icharg=1|icharg=11'):
                subprocess.Popen(['rsync', '-a', '-h', '--info=progress2', prev.path+'/CHG', path+'/CHG']).wait()
            if gen.parse_if('icharg=0|icharg=10|istart=1|istart=2'):
                subprocess.Popen(['rsync', '-a', '-h', '--info=progress2', prev.path+'/WAVECAR', path+'/WAVECAR']).wait()
            if prev and getattr(prev, 'vasp', None) and getattr(prev.vasp, 'optimized_cell', None):
                node.cell = copy.deepcopy(prev.vasp.optimized_cell)
                print self.__class__.__name__ + '.compute: prev.vasp.optimized_cell overwrites node.cell.'
            elif prev and getattr(prev, 'cell', None):
                node.cell = copy.deepcopy(prev.cell)
                print self.__class__.__name__ + '.compute: prev.vasp.cell overwrites node.cell.'
            # write incar etc. Relies on inheritance.
            os.chdir(path)
            gen.write_incar_kpoints()
            with open('POSCAR','w') as f:
                if gen.parse_if('mlpbsopt'):
                    print self.__class__.__name__ + '.compute: pre-optimizing cell. [MlPbSOpt]'
                    # cell = dynamic.services['MLPBSOPT'].optimize(node.cell)
                else:
                    cell = node.cell
                f.write(str(cell))
            for symbol in node.cell.stoichiometry.keys():
                gen.pot(symbol)
            # setting variables for wrapper
            ncore_total = str(  int(gen.getkw('nnode')) * int(gen.getkw('ncore_node'))  )
            if gen.parse_if('spin=ncl'):
                flavor = 'ncl'
            elif gen.getkw('kpoints').split()[0] in 'GM' and all([int(x)==1 for x in gen.getkw('kpoints').split()[1:]]):
                flavor = 'gam'
            else:
                flavor = 'std'
            self.remote_folder_name = ''.join(e for e in name if e.isalnum()) + ''.join(random.sample(string.ascii_lowercase,4))
            # write scripts and instructions
            # subfile actually runs vasp. wrapper submits the subfile to system.
            self.wrapper = '#!/bin/bash\n' ; self.subfile = '#!/bin/bash\necho $PWD `date` start\necho -------------------------\n'
            if gen.parse_if('platform=dellpc_gpu'):
                self.subfile += dedent('''\
                    (while true; do
                        echo `date +%%s` `nvidia-smi | sed -n '9p' | awk '{print $9}'` | sed 's/MiB//g' >> ./gpu.log
                        sleep 3
                    done) &
                    bgPID=$!
                    mpiexec.hydra -n %s /home/xzhang1/src/vasp.5.4.1/bin/vasp_gpu </dev/null
                    mail -s "VASP job finished: {${PWD##*/}}" 8576361405@vtext.com <<<EOM
                    kill "$bgPID"
                    ''' %(ncore_total))
                self.wrapper += 'nohup ./subfile 2>&1 >> run.log &'
            elif gen.parse_if('platform=dellpc'):
                self.subfile += dedent('''\
                    (while true; do
                        echo `date +%%s` `free -b | sed -n '2p' | awk '{print $3}'` >> ./cpu.log
                        sleep 3
                    done) &
                    bgPID=$!
                    mpiexec.hydra -n %s /home/xzhang1/src/vasp.5.4.1/bin/vasp_%s </dev/null
                    mail -s "VASP job finished: {${PWD##*/}}" 8576361405@vtext.com <<<EOM
                    kill "$bgPID"
                    ''' %(ncore_total, flavor))
                self.wrapper += 'nohup ./subfile 2>&1 >> run.log &'
            elif gen.parse_if('platform=nanaimo'):  # netspeed is ~50M/s. compression might slow it down.
                self.wrapper += dedent('''\
                    rsync -avP . nanaimo:~/%s
                    ssh nanaimo <<EOF
                      cd %s
                      sbatch --nodes=%s --ntasks=%s --job-name=%s -t 48:00:00 --export=ALL subfile
                    EOF
                    ''' %(self.remote_folder_name, self.remote_folder_name, gen.getkw('nnode'), ncore_total, self.remote_folder_name))
                self.subfile += dedent('''\
                    #!/bin/bash
                    . /usr/share/Modules/init/bash
                    module purge
                    module load intel
                    module load impi
                    mpirun -np %s /opt/vasp.5.4.4/bin/vasp_%s
                    ''' %(ncore_total, flavor))
            elif gen.parse_if('platform=irmik'):
                self.wrapper += dedent('''\
                    rsync -avP . irmik:~/%s
                    ssh irmik <<EOF
                     cd %s
                     sbatch --nodes=%s --ntasks=%s --job-name=%s -t 48:00:00 --export=ALL subfile
                    EOF
                    ''' %(self.remote_folder_name, self.remote_folder_name, gen.getkw('nnode'), ncore_total, self.remote_folder_name))
                self.subfile += dedent('''\
                    #!/bin/bash
                    # . /usr/share/Modules/init/bash
                    # module purge
                    # module load mvapich2-2.2/intel
                    # mpirun -np %%s /opt/vasp.5.4.4/bin/vasp_%%s
                    mpirun -np %s vasp_%s
                    ''' %(ncore_total, flavor))
            elif gen.parse_if('platform=nanaimo'):
                self.wrapper += dedent('''\
                    rsync -avP . nanaimo:~/%s
                    ssh nanaimo <<EOF
                      cd %s
                      sbatch --nodes=%s --ntasks=%s --job-name=%s -t 48:00:00 --export=ALL subfile
                    EOF
                    ''' %(self.remote_folder_name, self.remote_folder_name, gen.getkw('nnode'), ncore_total, self.remote_folder_name))
                self.subfile += dedent('''\
                    #!/bin/bash
                    . /usr/share/Modules/init/bash
                    module purge
                    module load intel
                    module load impi
                    mpirun -np %s /opt/vasp.5.4.4/bin/vasp_%s
                    ''' %(ncore_total, flavor))
            elif gen.parse_if('platform=edison'):
                self.wrapper += dedent('''\
                    rsync -avP . edison:~/%s
                    ssh edison <<EOF
                      cd %s
                      sbatch --nodes=%s --ntasks=%s --job-name=%s -t 36:00:00 --partition=regular subfile
                    EOF
                    ''' %(self.remote_folder_name, self.remote_folder_name, gen.getkw('nnode'), ncore_total, self.remote_folder_name))
                self.subfile += dedent('''\
                    #!/bin/bash
                    module load vasp
                    srun -n %s vasp_%s
                    ''' %(ncore_total, flavor))
            self.subfile += 'echo $PWD `date` end \necho -------------------------\n'
            with open('wrapper','w') as of_:
                of_.write(self.wrapper)
                os.system('chmod +x wrapper')
            if self.subfile:
                with open('subfile','w') as of_:
                    of_.write(self.subfile)
                os.system('chmod +x subfile')
            if shared.DEBUG <= 0:
                print self.__class__.__name__ + ': running wrapper'
                # these lines don't look right.
                # with open(os.devnull, 'r+b', 0) as DEVNULL:
                #     subprocess.Popen(['bash', './wrapper'], stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL, preexec_fn=os.setpgrp, close_fds=True)
                subprocess.Popen(['bash','./wrapper']).wait()
                print self.__class__.__name__ + ': computation started. local path   %s   . waiting for filesystem update. ' %path
            else:
                print '-' * 50 + '\n' + self.__class__.__name__ + ': wrapper generated at   %s   . waiting for filesystem update. ' %path


        # no log but invoked, only possible from moonphase. write parent.
        elif not getattr(self,'log',None):
            os.chdir(path)
            # download folder
            if gen.parse_if('platform=nanaimo|platform=irmik|platform=hodduk'):
                print '%s.compute: copying remote folder {%s} back to path {%s}' %(self.__class__.__name__, self.remote_folder_name, path)
                subprocess.Popen(['rsync', '-a', '-h', '--info=progress2', '%s:%s/' %(gen.getkw('platform'),self.remote_folder_name), '%s'%path]).wait()
                #os.system('scp -r /home/xzhang1/Shared/%s/%s/ %s' %(gen.getkw('platform'), self.remote_folder_name, path))
                print self.__class__.__name__ + '.compute: copy complete.'
            # log
            l = os.listdir(path)
            filename = ([x for x in l if x.startswith('slurm-')] + [x for x in l if x.startswith('run.log')] + [x for x in l if x.startswith('OSZICAR')])[0]
            with open(filename,'r') as if_:
                self.log = if_.read()
            # optimized_cell
            print self.__class__.__name__ + '.compute: recording vasp.optimized_cell'
            if gen.parse_if('opt'):
                with open('CONTCAR','r') as f_:
                    text = f_.read()
                    setattr(self, 'optimized_cell', Cell(text))
            # MLVASPSPEED
            # try:
            #     print self.__class__.__name__ + '.compute: collecting data for MLVASPSPEED'
            #     dynamic.services['MLVASPSPEED'].parse_train(node, self, gen, node.cell, Makeparam(gen))
            # except shared.CustomError as e:
            #     print(e)
            # print self.__class__.__name__ + '.compute: MLVASPSPEED collection complete'
            # # MLQUEUETIME
            # try:
            #     print self.__class__.__name__ + '.compute: collecting data for MLQUEUETIME'
            #     dynamic.services['MLQUEUETIME'].parse_train(node, self, gen)
            # except shared.CustomError as e:
            #     print(e)
            # print self.__class__.__name__ + '.compute: MLQUEUETIME collection complete'
            # MLPBSOPT
            # try:
            #     print self.__class__.__name__ + '.compute: collecting data for MLPBSOPT'
            #     dynamic.services['MLPBSOPT'].parse_train(self)
            # except shared.CustomError as e:
            #     print(e)
            # print self.__class__.__name__ + '.compute: MLPBSOPT collection complete'


        else:
            print self.__class__.__name__ + ' compute: calculation already completed at %s. Why are you here?' %path

    @shared.moonphase_wrap
    def moonphase(self):

        node = self.node()
        path = node.path
        gen = node.gen
        platform = gen.getkw('platform')
        rpath = path if platform=='dellpc' else self.remote_folder_name

        if shared.DEBUG>=2:    print 'calling %s(%s).moonphase' %(self.__class__.__name__, getattr(self,'path',''))

        if not getattr(self, 'wrapper', None):
            return 0

        elif not getattr(self, 'log', None):

            # implements the choke mechanism. instead of reporting computable, report choke. unless detects computation complete, then report success/fail

            '''
            If vasp is running, return 1.
            Otherwise, if vasprun.xml doesn't exist, return 1.
            Otherwise return
            '''
            is_running = self.info('is_running')

            output = ssh_and_run(platform=self.node().gen.getkw('platform'), command='[ -e %s/vasprun.xml ] && echo ok || echo notok' %rpath).splitlines()[0]
            vasprunxml_exist = output=='ok'

            if vasprunxml_exist:
                vasprunxml_lastline = ssh_and_run(platform=self.node().gen.getkw('platform'), command='tail -1 %s/vasprun.xml' %rpath).splitlines()[0]

            if is_running or not vasprunxml_exist:
                return 1
            elif '</modeling>' in vasprunxml_lastline:
                self.compute()
                return 2
            else:
                return -1

        else:   # getattr(self,'log',None)!=None
            return 2

    def delete(self):
        path = self.node().path
        if os.path.isdir(path):
            print 'removing self.path {%s}' %path
            os.chdir(shared.script_dir)
            subprocess.Popen(['trash', path]).wait()

    def __str__(self):
        #: return log and optimized_cell
        if getattr(self, 'optimized_cell', None):
            return '# optimized_cell:\n' + str(self.optimized_cell) + self.log
        if getattr(self, 'log', None):
            return self.log
        else:
            return 'moonphase is not 2, nothing here'

    def info(self, info=None):
        '''Accounting tool for collecting job status.'''
        node = self.node()
        path = node.path
        gen = node.gen
        platform = gen.getkw('platform')
        if info == 'is_running':
            if platform == 'dellpc' or platform == 'dellpc_gpu':
                try:
                    pgrep_output = check_output(['pgrep','vasp'])
                    return pgrep_output.strip() != ''
                except CalledProcessError:
                    return False
            elif platform in ['nanaimo', 'irmik', 'comet', 'edison', 'cori']:
                if shared.DEBUG>=2: print self.__class__.__name__ + '.moonphase: asking %s for status of {%s}' %(gen.getkw('platform'), path)
                result = ssh_and_run(platform=self.node().gen.getkw('platform'), command="squeue -n '%s'" %(self.remote_folder_name))
                return ( len(result.splitlines()) > 1 )
            else:
                raise shared.CustomError(self.__class__.__name__ + '.info: invalid platform')
        elif info == 'memory_used':
            if gen.parse_if('platform=dellpc_gpu'):
                if not os.path.exists(path + '/gpu.log'):
                    raise shared.CustomError(self.__class__.__name__ + '.info: no gpu.log')
                with open(path + '/gpu.log', 'r') as f:
                    l = np.float_([l.split() for l in f.readlines()])
                    return np.max(l[:,1]) - np.min(l[:, 1])
            elif gen.parse_if('platform=dellpc'):
                if not os.path.exists(path + '/cpu.log'):
                    raise shared.CustomError(self.__class__.__name__ + '.info: no cpu.log')
                with open(path + '/cpu.log', 'r') as f:
                    l = np.float_([l.split() for l in f.readlines()])
                    return np.max(l[:,1]) - np.min(l[:, 1])
            elif platform in ['nanimo', 'irmik']:
                # output = self.ssh_and_run("sacct -S 0101 -u xzhang1 --name=%s" %(self.remote_folder_name)).splitlines()
                raise shared.CustomError(self.__class__.__name__ + '.memory_used: for now, irmik and nanimo stats are unusable.')
            else:
                raise shared.CustomError(self.__class__.__name__ + '.info: invalid platform')
        elif info == 'run_time':
            if self.moonphase() != 2:
                raise shared.CustomError(self.__class__.__name__ + '.warning: vasp moonphase is not 2. skipped collect data.')
            if not os.path.isfile(node.path+'/OUTCAR'):
                raise shared.CustomError(self.__class__.__name__ + '.warning: no OUTCAR found. skipped collect data.')
            # parse outcar for time (s) / #elecstep
            os.chdir(node.path)
            with open(node.path + '/OUTCAR', 'r') as f:
                lines = f.readlines()
                # total time
                line = [l for l in lines if 'Total CPU time used' in l]
                if not line:
                    raise shared.CustomError(self.__class__.__name__ + '.warning: no Total CPU time line found. skipped collect data.')
                total_time = float(line[-1].split()[-1])
                if total_time < 1:
                    raise shared.CustomError(self.__class__.__name__ + '.warning: total time does not feel right. skipped collect data.')
                return total_time
        elif info == 'queue_time':
            if platform == 'dellpc' or platform == 'dellpc_gpu':
                raise shared.CustomError(self.__class__.__name__ + '.info: there is no queue time for dell. Yes, it can be 0, but bad for the data.')
            elif platform in ['nanaimo', 'irmik', 'comet', 'edison', 'cori']:
                output = ssh_and_run(platform=self.node().gen.getkw('platform'), command="sacct -S 0101 --format=jobname%%100,submit,start -u xzhang1 --name=%s" %(self.remote_folder_name)).splitlines()
                lines = [l.split() for l in output]
                submit = next(l for l in lines if l[0]==self.remote_folder_name)[1]
                submit = time.mktime(dateparser.parse(submit).timetuple())
                start = next(l for l in lines if l[0]==self.remote_folder_name)[2]
                start = time.mktime(dateparser.parse(start).timetuple())
                now = time.mktime(dateparser.parse('now').timetuple())
                # if start < now - 5184000:
                #     raise shared.CustomError('Data is over 2 months old. ')
                if start - submit < 10:
                    raise shared.CustomError(self.__class__.__name__ + '.info: no wait time - good news, but not what I need')
                return start - submit
            else:
                raise shared.CustomError(self.__class__.__name__ + '.info: invalid platform')
        elif info == 'n_electronic_step':
            if self.moonphase() != 2:
                raise shared.CustomError(self.__class__.__name__ + '.warning: vasp moonphase is not 2. skipped collect data.')
            if not os.path.isfile(node.path+'/OUTCAR'):
                raise shared.CustomError(self.__class__.__name__ + '.warning: no OUTCAR found. skipped collect data.')
            # parse outcar for time (s) / #elecstep
            os.chdir(node.path)
            with open(node.path + '/OUTCAR', 'r') as f:
                iteration_lines = [l for l in f.readlines() if 'Iteration' in l]
                return len(iteration_lines)
        elif info == 'n_ionic_step':
            if not os.path.exists(path + '/OSZICAR'):
                return None
            with open(path + '/OSZICAR', 'r') as f:
                line = [l for l in f.readlines() if 'F=' in l]
                return len(line)


#===========================================================================


class Dummy(object):

    def __init__(self, node):
        self.path = node.path
        self.gen = node.gen

    def compute(self):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
            self.log = 'Made directory %s' % (self.path)
        else:
            self.log = 'Identified directory %s' % (self.path)

    @shared.moonphase_wrap
    def moonphase(self):
        return 2 if getattr(self, 'log', None) else 0

    def __str__(self):
        return self.log if self.moonphase()==2 else 'moonphase is not 2, nothing here'

    def delete(self):
        print 'removing folder {%s}' %self.path
        if os.path.isdir(self.path):
            os.chdir(shared.script_dir)
            shutil.rmtree(self.path)


#===========================================================================

# Electron


class Grepen(object):


    @shared.log_wrap
    def __init__(self, electron):

        prev = Map().rlookup(attr_dict={'electron':electron}, prev=True)

        self.energy=float(os.popen('grep "energy without" OUTCAR | tail -1 | awk \'{print $5}\'').read())
        self.efermi=float(os.popen('grep "E-fermi" OUTCAR | tail -1 | awk \'{print $3}\'').read())

        self.nbands=int(os.popen('grep NBANDS OUTCAR | awk \'{print $15}\'').read())
        self.nedos=int(os.popen('grep NEDOS OUTCAR | awk \'{print $6}\'').read())

        self.spin = prev.gen.getkw('spin')
        self.ismear = int(prev.gen.getkw('ismear'))
        self.sigma = 0 if self.ismear!=0 else float(prev.gen.getkw('ismear'))

        with open('DOSCAR','r') as doscar_file:
            self.is_doscar_usable = len(doscar_file.readlines()) > 7

        with open('KPOINTS','r') as kpoints_file:
            kpoints = prev.gen.getkw('kpoints').split()
            self.is_kpoints_mesh = ( kpoints[0] in 'GM' and np.prod(np.float32(kpoints[1:]))>4 and len(kpoints_file.readlines())<7 )

        with open("EIGENVAL","r") as eigenval_file:
            eigenval = [ x.split() for x in eigenval_file.readlines() ]
            self.temperature = float( eigenval[2][0] )
            self.nelectrons = int( eigenval[5][0] )
            self.nkpts = int( eigenval[5][1] )
            if self.nkpts != (len(eigenval) - 6) / (self.nbands+2):
                raise shared.CustomError(self.__class__.__name__ + '__init__: EIGENVAL file length not matching nkpts.')

        for name, value in vars(self).iteritems():
            if name != 'log':
                print '%s: %s\n' % (name, value)



class Dos(object):

    @shared.MWT(timeout=2592000)
    def dos_interp(self):
        energy = 0 ; DOS = 1    # dict-like indexing: a = [3eV, 0.28]; print a[energy]
        return [ interp1d( self.dos[idx_spin, :, energy], self.dos[idx_spin, :, DOS] ) \
                 for idx_spin in range(self.nspins_dos) ]

    @shared.MWT(timeout=2592000)
    def pdos_interp(self):
        energy = 0 ; PDOS = 1
        return [ [ [ interp1d(self.pdos[idx_spin, idx_atom, idx_orbital, :, energy], self.pdos[idx_spin, idx_atom, idx_orbital, :, PDOS], kind='cubic') \
                     for idx_orbital in range(self.norbitals_pdos) ] \
                        for idx_atom in range(sum(self.node().cell.stoichiometry.values())) ] \
                            for idx_spin in range(self.nspins_pdos) ]

    @shared.log_wrap
    def __init__(self, electron):
        ZERO = 0.001
        self.nspins_dos = {'para':1, 'fm':2, 'ncl':1}[electron.grepen.spin]
        self.nspins_pdos = {'para':1, 'fm':2, 'ncl':4}[electron.grepen.spin]
        for name in ['nspins_dos', 'nspins_pdos']:
            print '%s: %s\n' % (name, getattr(self, name, None))
        print '-' * 130

        if not electron.grepen.is_doscar_usable:
            raise shared.CustomError(self.__class__.__name__ + '.__init__: DOSCAR is not usable.')

        # all DOSCAR lines, including dos and pdos
        with open("DOSCAR","r") as f:
            doscar = [ line.split() for line in f.readlines() ]
            doscar_ifs = doscar[5]
            doscar_parts = [list(doscar_part) for b, doscar_part in groupby(doscar, lambda x,doscar_ifs=doscar_ifs: x==doscar_ifs or not x) if not b]
            doscar_tot = doscar_parts[1]
            doscar_site = doscar_parts[3:]  # doscar_site[idx_atom=0]

        # dos
        self.dos = np.zeros([self.nspins_dos, len(doscar_tot), 2])
        for idx, doscar_tot_ in enumerate(doscar_tot):
            #
            self.dos[:, idx, 0] = doscar_tot_.pop(0)
            #
            for idx_spin in range(self.nspins_dos):
                self.dos[idx_spin, idx, 1] = doscar_tot_.pop(0)
        self.idx_fermi = abs(self.dos[0, :, 0] - electron.grepen.efermi).argmin() + 1
        # bandgap
        self.bandgap = [ [] for idx_spin in range(self.nspins_dos) ]
        for idx_spin in range(self.nspins_dos):
            i = self.idx_fermi
            while abs(self.dos[idx_spin, i, 1]) < ZERO:
                i -= 1
            j = self.idx_fermi
            while abs(self.dos[idx_spin, j, 1]) < ZERO:
                j += 1
            self.bandgap[idx_spin] = [] if i == j else [self.dos[idx_spin, i, 0], self.dos[idx_spin, j, 0]]
            print "spin %s: VBM %s, CBM %s, bandgap %.5f eV\n" % (idx_spin, self.bandgap[idx_spin][0], self.bandgap[idx_spin][1], self.bandgap[idx_spin][1]-self.bandgap[idx_spin][0]) \
                  if i != j else "spin %s: no bandgap\n" % (idx_spin)

        # pdos
        self.norbitals_pdos = ( len(doscar_site[0][0]) - 1 ) / self.nspins_pdos
        self.pdos = np.zeros([self.nspins_pdos, sum(electron.cell.stoichiometry.values()), self.norbitals_pdos, len(doscar_site[0]), 2])
        for idx_atom, doscar_site_atom in enumerate(doscar_site):
            for idx, doscar_site_atom_ in enumerate(doscar_site_atom):
                #
                self.pdos[:, idx_atom, :, idx, 0] = doscar_site_atom_.pop(0)
                #
                for idx_orbital in range(self.norbitals_pdos):
                    for idx_spin in range(self.nspins_pdos):
                        self.pdos[idx_spin, idx_atom, idx_orbital, idx, 1] = doscar_site_atom_.pop(0)




# imports bandstructure from EIGENVAL.
# imports kpoints list.
# interpolates.
# finds all sources of errors in bandstructure.
class Bands(object):

    @shared.MWT(timeout=2592000)
    def bands_interp(self):
        '''fit the band for verifying smoothness, and interpolating bandgap'''
        return [ [ Rbf(self.kpts[:,0], self.kpts[:,1], self.kpts[:,2], self.bands[idx_spin, idx_band])
                           for idx_band in range(self.grepen.nbands) ] for idx_spin in range(self.nspins_bands) ]


    @shared.log_wrap
    def __init__(self, electron):
        ZERO = 0.01
        self.nspins_bands = {'para':1, 'fm':2, 'ncl':1}[electron.grepen.spin]
        print 'nspins_bands: %s\n' % (self.nspins_bands)
        print '-' * 130

        # bands
        with open("EIGENVAL","r") as f:
            eigenval = [x.split() for x in f.readlines()]
        self.kpts = np.zeros([electron.grepen.nkpts,3])
        self.bands = np.zeros([self.nspins_bands, electron.grepen.nbands, electron.grepen.nkpts])
        # pop header lines
        del eigenval[:6]
        for idx_kpt in trange(electron.grepen.nkpts, leave=False, desc='reading bands'):
            # pop [empty line]
            eigenval.pop(0)
            eigenval_ = eigenval.pop(0)
            self.kpts[idx_kpt,:] = eigenval_[:3]
            #
            for idx_band in range(electron.grepen.nbands):
                eigenval_ = eigenval.pop(0) ; eigenval_.pop(0)
                for idx_spin in range(self.nspins_bands):
                    self.bands[idx_spin, idx_band, idx_kpt] = eigenval_.pop(0)

        # bandgap
        self.bandgaps = [ [] for idx_spin in range(self.nspins_bands) ]
        for idx_spin in range(self.nspins_bands):
            vbm = max([e for e in tqdm(np.nditer(self.bands[idx_spin]), leave=False) if e<=electron.grepen.efermi + ZERO])    # else VB slightly beyond efermi is considered CB
            cbm = min([e for e in tqdm(np.nditer(self.bands[idx_spin]), leave=False) if e>=electron.grepen.efermi + ZERO])    # np.nditer is an iterator looping over all dimensions of an array.
                                                                                               # the array itself is an iterator looping normally by outmost dimension.
            self.bandgaps[idx_spin] = [vbm, cbm] if cbm > vbm + ZERO else []
            print "spin %s: VBM %s at %s, CBM %s at %s, kpoint-independent bandgap %.5f eV\n" \
                  % (idx_spin, vbm, self.kpts[ np.where(self.bands[idx_spin]==vbm)[1][0] ], cbm, self.kpts[ np.where(self.bands[idx_spin]==cbm)[1][0] ], cbm-vbm) \
                  if cbm > vbm + ZERO else "spin %s: no bandgap\n" % (idx_spin)     # only first instance is printed.
        print '-' * 130

        # interpolated bandgap
        print 'bandgap is often between interpolated and raw bandgap.'
        if electron.grepen.is_kpoints_mesh:
            self.bandgaps_interp = [ [] for idx_spin in range(self.nspins_bands) ]
            self.min_kpt_dist = np.amin( spatial.distance.pdist(self.kpts, metric='Euclidean'), axis=None )   # spatial.distance.pdist() produces a list of distances. amin() produces minimum for flattened input
            # optimize for each spin and each band
            for idx_spin in range(self.nspins_bands):
                ZERO = 0.01
                if not self.bandgaps[idx_spin]: # conductor
                    print 'spin %s: no bandgap, bandgaps_interp skipped.\n' % ( idx_spin ) ; continue
                if [idx2_spin for idx2_spin in range(idx_spin) if self.bandgaps[idx2_spin] and np.linalg.norm(np.subtract(self.bandgaps[idx_spin], self.bandgaps[idx2_spin])) < ZERO]:    # repetitive
                    print 'spin %s: repetitive, bandgaps_interp skipped.\n' % ( idx_spin ) ; continue
                #
                kptes = []
                ZERO = abs(np.subtract(*self.bandgaps[idx_spin])) / 2.5
                for idx_band in trange(electron.grepen.nbands, leave=False, desc='interpolating bands for bandgap', position=0):
                    if any(self.bandgaps[idx_spin][0] - ZERO < e < self.bandgaps[idx_spin][1] + ZERO for e in self.bands[idx_spin, idx_band]):
                        for kpt in tqdm(self.kpts, leave=False, position=1):
                            for sign in (-1,1):
                                e = sign * scipy.optimize.minimize(lambda x,self=self,idx_spin=idx_spin,idx_band=idx_band,sign=sign: self.bands_interp()[idx_spin][idx_band](*x) * sign,
                                                                   x0 = kpt,
                                                                   bounds = [[x-self.min_kpt_dist*0.5,x+self.min_kpt_dist*0.5] for x in kpt],
                                                                   tol=1e-6).fun
                                kptes.append([kpt[0],kpt[1],kpt[2],e])
                kptes = np.float32(kptes)
                # self.bandgaps_interp
                vbm = np.amax([kpte[3] for kpte in kptes if self.bandgaps[idx_spin][0]-ZERO<kpte[3]<self.bandgaps[idx_spin][0]+ZERO])
                cbm = np.amin([kpte[3] for kpte in kptes if self.bandgaps[idx_spin][1]-ZERO<kpte[3]<self.bandgaps[idx_spin][1]+ZERO])
                self.bandgaps_interp[idx_spin] = [vbm, cbm] if cbm>vbm else []
                print "spin %s, interpolated: VBM %s near %s , CBM %s near %s, bandgap %.5f eV\n" \
                      % (idx_spin, vbm, kptes[np.where(kptes[:,3]==vbm)[0][0],:3], cbm, kptes[np.where(kptes[:,3]==cbm)[0][0],:3], cbm-vbm) \
                      if cbm > vbm else "spin %s, interpolated: no bandgap\n" % (idx_spin)

    def plot(self, idx_spin, idx_band):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs, ys, zs, cs = self.kpts[:,0], self.kpts[:,1], self.kpts[:,2], self.bands[idx_spin,idx_band]
        p = ax.scatter(xs, ys, zs, s=15, c=cs)
        #
        ax.set_xlabel('KX')
        ax.set_ylabel('KY')
        ax.set_zlabel('KZ')
        #
        fig.colorbar(p)
        #
        plt.show()
        return


# executes population analysis
class Charge(object):


    @shared.log_wrap
    def __init__(self, electron):
        if electron.dos.norbitals_pdos > len(shared.ELEMENTS.orbitals):
            raise shared.CustomError(self.__class__.__name__ +' __init__: more orbitals than supported.')

        # let's start!
        # pristine electronic configuration
        print 'ground-state electron configurations:\n'   # a good idea is to indent within header
        for element in electron.cell.stoichiometry:
            print "%s: %s\n" %(element, shared.ELEMENTS[element].eleconfig)
        print '-' * 130

        # integrate pdos scaled
        print 'Integrated PDOS. Each orbital is normalized to 1. If ONE is too small, -INT is returned.\n'
        idx_atom = 0
        for symbol, natoms in electron.cell.stoichiometry.iteritems():
            for idx_atom in range(idx_atom, idx_atom + natoms):
                for idx_spin in range(electron.dos.nspins_pdos):
                    print "%5s%2s: " % ( symbol + str(idx_atom), shared.ELEMENTS.spins[electron.dos.nspins_pdos][idx_spin] )
                    for idx_orbital in range(electron.dos.norbitals_pdos):
                        INFINITY = np.argmax( electron.dos.pdos[idx_spin, idx_atom, idx_orbital, :, 0 ] > electron.grepen.efermi+5 )
                        ONE = np.trapz( electron.dos.pdos[idx_spin, idx_atom, idx_orbital, :INFINITY, 1 ] , \
                                                    x = electron.dos.pdos[idx_spin, idx_atom, idx_orbital, :INFINITY, 0 ] )
                        integrated_pdos = np.trapz( electron.dos.pdos[idx_spin, idx_atom, idx_orbital, :electron.dos.idx_fermi, 1 ] , \
                                                    x = electron.dos.pdos[idx_spin, idx_atom, idx_orbital, :electron.dos.idx_fermi, 0 ] )
                        print '%7s %5.2f' % (shared.ELEMENTS.orbitals[idx_orbital], abs(integrated_pdos / ONE) if abs(ONE) > 0.4 else -abs(integrated_pdos) )
                    print '\n'
        print '-' * 130

        # Bader charge
        print "\n\nBader charge. Boundaries are defined as zero-flux surfaces. Note that certain flags should be set (e.g. LAECHG) for this to be reasonable.\n"
        print 'running bader...', ; sys.stdout.flush()
        if not os.path.isfile('AECCAR0') or not os.path.isfile('AECCAR2'):
            raise shared.CustomError(self.__class__.__name__ + '.__init__: no AECCAR found. You forgot to add LAECHG-tag. Bader will not run reliably.')
        os.popen('chgsum.pl AECCAR0 AECCAR2').read()
        os.popen('bader CHGCAR_sum').read()
        print '\r                 \r', ; sys.stdout.flush()
        with open('ACF.dat','r') as f:
            lines = f.readlines()
        for idx_element, element in enumerate(electron.cell.stoichiometry.keys()):
            print element + ' '
            for idx_atom in range(sum(electron.cell.stoichiometry.values()[0:idx_element]),sum(electron.cell.stoichiometry.values()[0:idx_element+1])):
                nline = idx_atom + 2
                print lines[nline].split()[4] + ' '
            print '\n'
        print '-' * 130

        # OUTCAR RWIGS [decomposed] charge
        print "Total charge inside the Wigner-Seitz Radius in OUTCAR\n"
        with open('OUTCAR','r') as f:
            lines = f.readlines()
            for idx_line,line in enumerate(lines):
                if '# of ion' in line: # this is only an anchor. that particular line doesn't matter.
                    break
            for idx2_line in range(idx_line ,idx_line + 4 + sum(electron.cell.stoichiometry.values())):
                print lines[idx2_line]
        print '-' * 130

        # Bader magnetic moment
        if electron.grepen.spin == 'fm' or electron.grepen.spin == 'afm':
            print "An oversimplified version of Bader magnetic moment.\n"
            os.popen('chgsplit.sh CHGCAR').read()
            os.popen('bader cf2').read()
            for idx_element, element in enumerate(electron.cell.stoichiometry.keys()):
                print element + ' '
                for idx_atom in range(sum(electron.cell.stoichiometry.values()[0:idx_element]),sum(electron.cell.stoichiometry.values()[0:idx_element+1])):
                    nline = idx_atom + 2
                    with open('ACF.dat','r') as f:
                        lines = f.readlines()
                        print lines[nline].split()[4]
                print '\n'
        print '-' * 130

        # OUTCAR RWIGS magnetic moment
        if electron.grepen.spin == 'fm' or electron.grepen.spin == 'afm':
            print "Magnetization (x) [total magnetic moment inside the Wigner-Seitz Radius] in OUTCAR\n"
            with open('OUTCAR','r') as f:
                lines = f.readlines()
                for idx_line,line in enumerate(lines):
                    if 'magnetization (x)' in line: # this is only an anchor. that particular line doesn't matter.
                        break
                for idx2_line in range(idx_line + 2,idx_line + 6 + sum(electron.cell.stoichiometry.values())):
                    print lines[idx2_line]



class Errors(object):


    @shared.log_wrap
    def __init__(self, electron):

        self.error = 0

        # delta_k
        if electron.grepen.is_kpoints_mesh:
            self.min_kpt_dist = np.amin( spatial.distance.pdist(electron.bands.kpts, metric='Euclidean'), axis=None )   # spatial.distance.pdist() produces a list of distances. amin() produces minimum for flattened input
            kpts_nn = spatial.cKDTree( electron.bands.kpts )                                                        # returns a KDTree object, which has interface for querying nearest neighbors of any kpt
            kpts_nn_list = kpts_nn.query_pairs(r=self.min_kpt_dist*1.5, output_type='ndarray')           # gets all nearest-neighbor idx_kpt pairs
            print u"kpoint mesh precision \u0394k = %.5f 2\u03C0/a.\n" %(self.min_kpt_dist)

        # neargap bands, de_dkpt
        # calculate DeltaE_KPOINTS by grabbing average E diff / average E diff near bandgap from EIGENVAL.
        # for each spin
        if electron.grepen.is_kpoints_mesh:
            for idx_spin in range(electron.bands.nspins_bands):
                ZERO = 0.01
                if not electron.bands.bandgaps[idx_spin]: # conductor
                    print u'spin %s: no bandgap, \u3B4E skipped.\n' % idx_spin ; continue
                if [idx2_spin for idx2_spin in range(idx_spin) if electron.bands.bandgaps[idx2_spin] and np.linalg.norm(np.subtract(electron.bands.bandgaps[idx_spin], electron.bands.bandgaps[idx2_spin])) < ZERO]:    # repetitive
                    print u'spin %s: repetitive, \u3B4E skipped.' % ( idx_spin ) ; continue
                # specify neargap criterion ZERO
                print u'spin %s, nearest neighbor \u03B4E:\n' % (idx_spin)
                bandgap = abs(np.subtract(*electron.bands.bandgaps[idx_spin]))
                for ZERO in tqdm([bandgap, bandgap/2, bandgap/4], leave=False, position=0, desc='calculating de_dkpt in neargap bands'):
                    de_dkpt_flat = []
                    # for each NN pair, compute |de_dkpt| if energy is within bound
                    for idx_band in trange(grepen.nbands, leave=False, position=1):
                        for kpts_nn_list_ in kpts_nn_list:  # kpts_nn_list_ = [ idx1_kpt idx2_kpt ]
                            if all(electron.bands.bandgaps[idx_spin][0]-ZERO < electron.bands.bands[idx_spin][idx_band][idx_kpt] < electron.bands.bandgaps[idx_spin][1]+ZERO for idx_kpt in kpts_nn_list_):    # is near gap
                                de_dkpt_flat.append( abs(electron.bands.bands[idx_spin][idx_band][kpts_nn_list_[0]] - electron.bands.bands[idx_spin][idx_band][kpts_nn_list_[1]]) )
                    print u'  CBM/VBM +- %.2f eV: \u03B4E = %.5f eV, # of kpts = %d.\n' %( ZERO, np.mean(de_dkpt_flat), len(de_dkpt_flat) ) \
                                if de_dkpt_flat else u'  CBM/VBM +- %.2f eV: # of kpts = 0.\n' %( ZERO )
                    if de_dkpt_flat :   self.de_dkpt = np.mean(de_dkpt_flat)    # for Errors
            print '-' * 130

        # ismear -> error
        if electron.grepen.ismear == 0:
            print u'gaussian smearing smoothes out irregularities with size sigma: sigma[%.4f] < \u03B4E[%.4f]/2\n' %(electron.grepen.sigma,self.error)
            self.error = max(self.error, electron.grepen.sigma * 2)

        # de_dkpt -> error
        if electron.grepen.is_kpoints_mesh:
            print u'sparse kpoints grid may miss in-between eigenvalues. E(j)-E(j\')[%.4f] < \u03B4E[%.4f]/2\n' %(self.de_dkpt, self.error)
            self.error = max(self.error, self.de_dkpt * 2)

        # nedos -> error
        if electron.grepen.is_doscar_usable:
            print u'all details between two DOS points are lost. 10/NEDOS[%.4f] < \u03B4[%.4f]/2\n' %(10.0/electron.grepen.nedos, self.error)
            self.error = max(self.error, 10.0/float(electron.grepen.nedos) * 2)

        # check DOS validity (nedos)
        if electron.grepen.is_doscar_usable:
            print u'dos should not be so fine that energy granularity is obvious. 10/NEDOS[%.4f] > \u03B4E[%.4f]\n' %(10.0/electron.grepen.nedos, self.de_dkpt if electron.grepen.is_kpoints_mesh else np.mean(np.diff(electron.bands.bands[0].flatten())))

        #: wrap-up
        print '-' * 130
        print 'errors.py: in short, you should expect an error around %.4f eV in dirA.' %(self.error)



class Electron(Dummy):
    '''the big boss'''

    def compute(self):

        prev = Map().rlookup(attr_dict={'electron':self}, prev=True)

        if not getattr(self, 'log', None):
            if os.path.isdir(self.path):
                raise shared.CustomError(self.__class__.__name__ + ' compute: self.path {%s} taken' %self.path)

            if self.gen.parse_if('cell'):

                print self.__class__.__name__ + '.compute: using previous path'
                os.chdir(prev.path)

                with open('POSCAR','r') as infile:
                    self.cell = Cell(infile.read())

            if self.gen.parse_if('grepen'):
                self.grepen = Grepen(self)

            if self.gen.parse_if('dos'):
                self.dos = Dos(self)

            if self.gen.parse_if('bands'):
                self.bands = Bands(self)

            if self.gen.parse_if('charge'):
                self.charge = Charge(self)

            if self.gen.parse_if('errors'):
                self.errors = Errors(self)

            self.log = ''
            for name in ['cell', 'grepen', 'dos', 'bands', 'charge', 'errors']:  # you've got to change this every time.
                if getattr(self, name, None) and getattr(getattr(self, name),'log', None):
                    print getattr(getattr(self, name), 'log').encode('utf-8')

        else:
            raise shared.CustomError(self.__class__.__name__ + ' compute: moonphase is 2. why are you here?')



# =============

# Compare

def compare_cell_bijective(eoc, boc, suppress_output = False):
    import itertools
    import numpy as np
    import os

    if suppress_output:
        orig_sys_stdout = sys.stdout
        sys.stdout = open(os.devnull,"w")

    # bijective-representation difference (congruent testing), allowing rotation and translation
    b = np.float32([ [i, j, np.linalg.norm(boc.ccoor[i]-boc.ccoor[j])] for i in range(boc.natoms()) for j in range(boc.natoms()) if i!=j ])
    e = np.float32([ [i, j, np.linalg.norm(eoc.ccoor[i]-eoc.ccoor[j])] for i in range(eoc.natoms()) for j in range(eoc.natoms()) if i!=j ])
    print u'<fixed-order bijective-representation difference> (allowing translation and rotation) between self and backdrop is: '
    print u'    avg difference: %.2f %%,  %.4f A.' %(np.divide(abs(b-e)[:,2], abs(b)[:,2]).mean() * 100, abs(b-e)[:,2].mean())
    result = np.divide(abs(b-e)[:,2], abs(b)[:,2]).mean() * 100
    idx_min = np.abs(b-e)[:,2].argmin()
    print u'    min difference: backdrop_pdist [%2d%s-%2d%s=%.3f] - electron_pdist [%2d%s-%2d%s=%.3f] = %.2f %%. ' %(b[idx_min][0], boc.ccoor[int(b[idx_min][0])], b[idx_min][1], boc.ccoor[int(b[idx_min][1])], b[idx_min,2],
                                                                                                                                   e[idx_min][0], eoc.ccoor[int(e[idx_min][0])], e[idx_min][1], eoc.ccoor[int(e[idx_min][1])], b[idx_min,2],
                                                                                                                                   np.abs(b-e)[idx_min,2] / np.abs(b)[idx_min,2] * 100)
    idx_max = abs(b-e)[:,2].argmax()
    print u'    max difference: backdrop_pdist [%2d%s-%2d%s=%.3f] - electron_pdist [%2d%s-%2d%s=%.3f] = %.2f %%. ' %(b[idx_max][0], boc.ccoor[int(b[idx_max][0])], b[idx_max][1], boc.ccoor[int(b[idx_max][1])], b[idx_max,2],
                                                                                                                                   e[idx_max][0], eoc.ccoor[int(e[idx_max][0])], e[idx_max][1], eoc.ccoor[int(e[idx_max][1])], e[idx_max,2],
                                                                                                                                   np.abs(b-e)[idx_max,2] / np.abs(b)[idx_max,2] * 100)
    print '-' * 130

    # bijective-representation difference (congruent testing), allowing physical phenomena relocation
    b = b[ b[:,2].argsort() ]
    e = e[ e[:,2].argsort() ]
    print u'<arbitrary-order bijective-representation difference> (allowing physical phenomena relocation) between self and backdrop is: '
    print u'    avg difference: %.2f %%,  %.4f A.' %(np.divide(abs(b-e)[:,2], abs(b)[:,2]).mean() * 100, abs(b-e)[:,2].mean())
    idx_min = np.abs(b-e)[:,2].argmin()
    print u'    min difference: backdrop_pdist [%2d(%s)-%2d(%s)=%.3f] - electron_pdist [%2d(%s)-%2d(%s)=%.3f] = %.2f %%. ' %(b[idx_min][0], boc.ccoor[int(b[idx_min][0])], b[idx_min][1], boc.ccoor[int(b[idx_min][1])], b[idx_min,2],
                                                                                                                                   e[idx_min][0], eoc.ccoor[int(e[idx_min][0])], e[idx_min][1], eoc.ccoor[int(e[idx_min][1])], e[idx_min,2],
                                                                                                                                   np.abs(b-e)[idx_min,2] / np.abs(b)[idx_min,2] * 100)
    idx_max = abs(b-e)[:,2].argmax()
    print u'    max difference: backdrop_pdist [%2d(%s)-%2d(%s)=%.3f] - electron_pdist [%2d(%s)-%2d(%s)=%.3f] = %.2f %%. ' %(b[idx_max][0], boc.ccoor[int(b[idx_max][0])], b[idx_max][1], boc.ccoor[int(b[idx_max][1])], b[idx_max,2],
                                                                                                                                   e[idx_max][0], eoc.ccoor[int(e[idx_max][0])], e[idx_max][1], eoc.ccoor[int(e[idx_max][1])], e[idx_max,2],
                                                                                                                                   np.abs(b-e)[idx_max,2] / np.abs(b)[idx_max,2] * 100)
    print '-' * 130

    if suppress_output:
        sys.stdout = orig_sys_stdout

    return result



def compare_cell(eoc,boc, ZERO=0.02, rs=[10, 6.5, 6.5], is_verbose=False):    # ZERO: relative difference. rs: 子簇大小。
    '''
    eoc就是新的cell。
    新的cell里面，原子根据自身的环境，感应到旧cell的相应位置，一个一个的亮了起来，形成一簇。
        加入已有的簇，必须与已有簇的位置匹配，而不论出身。
        当然，全新的簇就需要与未知簇的位置做无序匹配。
    有子簇分裂出去，那就是第二个core。

    （环境，要求局部化，免疫平移旋转，只能用pdist。）

    "What you want probably does not exist on this earth. --hackhands."
    '''
    import itertools
    import numpy as np
    report=''
    sreport=''

    cores = []  #eoc|boc, idx_core_atom
    remainder = [range(eoc.natoms()),range(boc.natoms())]

    # performance upgrade code
    eoc_cdist = eoc.cdist()
    boc_cdist = boc.cdist()

    for r in rs:    # 附录：考虑remainder里面分为多个派系的情况。做全排列太慢，只能进行局域性讨论。
        if is_verbose:  print '-' * 65 + ' r = %s ' %r + '-' * 65
        core = [[],[]]
        is_remainder_edited = True
        while is_remainder_edited and remainder[0] and remainder[1]:
            is_remainder_edited = False
            for idx_eoc in remainder[0]:        # 对照
                boc_winners = []
                for idx_boc in remainder[1]:
                    # 比较已经对照出的core
                    eoc_dist1 = eoc_cdist[idx_eoc,core[0]] if core[0] else [0]      ## 意外情况修复
                    boc_dist1 = boc_cdist[idx_boc,core[1]] if core[1] else [0]
                    # 比较未对照出的remainder，仅限r半径内
                    eoc_dist2_adjidx = [idx_atom for idx_atom in remainder[0] if 0<eoc_cdist[idx_atom,idx_eoc]<r]
                    eoc_dist2 = eoc_cdist[idx_eoc,eoc_dist2_adjidx]
                    boc_dist2_adjidx = [idx_atom for idx_atom in remainder[1] if 0<boc_cdist[idx_atom,idx_boc]<r]
                    boc_dist2 = boc_cdist[idx_boc,boc_dist2_adjidx]
                    l = min(len(eoc_dist2), len(boc_dist2))
                    # 选举最match的
                    ## 意外情况：core是空的（不空修复），remainder在r半径内是空的（delta2最终修复），分母有0（允许np.infty），分子分母同时为0（perfect match赋0）
                    with np.errstate(divide='ignore', invalid='ignore'):
                        delta1 = np.divide(np.float32(eoc_dist1)-np.float32(boc_dist1), np.float32(boc_dist1))
                        delta1[np.isnan(delta1)] = 0    ## 意外情况修复
                        delta1 = np.max(np.abs(delta1))
                        delta2 = np.divide(np.float32(eoc_dist2[:l])-np.float32(boc_dist2[:l]), np.float32(boc_dist2[:l]))
                        delta2[np.isnan(delta2)] = 0
                        delta2 = np.max(np.abs(delta2)) if delta2.size else 0 if core[0] else np.infty
                        boc_winners.append([idx_eoc, idx_boc, delta1, delta2])
                boc_winners = np.float32(boc_winners)
                if np.all(boc_winners[:,2] > ZERO):
                    if is_verbose:  print 'idx_eoc = %s: core match not found.' %idx_eoc
                elif np.all(boc_winners[:,3] > ZERO):
                    if is_verbose:  print 'idx_eoc = %s: remainder match not found.' %idx_eoc
                else:
                    idx_winnerslist = np.argmin(boc_winners[:,2] + boc_winners[:,3])
                    idx_eoc_winner = boc_winners[idx_winnerslist][0]
                    idx_boc_winner = boc_winners[idx_winnerslist][1]
                    core[0].append(int(idx_eoc_winner))
                    core[1].append(int(idx_boc_winner))
                    remainder[0].remove(int(idx_eoc_winner))
                    remainder[1].remove(int(idx_boc_winner))
                    is_remainder_edited = True
                    break
        cores.append(core)

    print '-' * 60 + ' max allowed relative difference = %s '%ZERO + '-' * 60
    print 'cores: %s' %(cores)
    print '-' * 60
    print 'remainder: %s' %(remainder)
    return report




class Compare(Dummy):

    @shared.log_wrap
    def compute(self):

        enode = Map().rlookup(attr_dict={'compare':self}, prev=True)
        bnode = Map().rlookup(attr_dict={'compare':self}, prev2=True)
        print 'compare experimental node {%s} against backdrop node {%s}' %(enode.name if enode else None, bnode.name if bnode else None)

        if self.gen.parse_if('etype=bands'):    # etype=btype=bands

            eelectron = enode.electron
            belectron = bnode.electron

            # preliminary checks
            print '-' * 130
            if not eelectron.grepen.is_kpoints_mesh and belectron.grepen.is_kpoints_mesh:
                raise shared.CustomError(self.__class__.__name__ + '.compute: only these: i) both mesh ii) neither mesh iii) backdrop only mesh.')
            if eelectron.bands.bands.shape[0] != blectron.bands.bands.shape[0]:
                raise shared.CustomError(self.__class__.__name__ + '.compute: experiment and backdrop bands have incompatible NBANDS')

            print u'energy difference between self and belectron is %s eV.\n' % ( abs(eelectron.grepen.energy - belectron.grepen.energy) )

            if eelectron.grepen.is_kpoints_mesh and belectron.grepen.is_kpoints_mesh:

                error_interpd = []
                idx_spin = 0
                print self.__class__.__name__ + ': comparing idx_spin=0 only.\n'    # faciliate comparison between ncl and fm

                for idx_band, band in tqdm(enumerate(belectron.bands.bands[idx_spin]), leave=False, desc='interpolating belectron.bands for comparison'):
                        error_interpd.append( np.average( abs( np.float32(eelectron.bands.bands[idx_spin][idx_band]) - np.float32([belectron.bands.bands_interp()[idx_spin][idx_band](*kpt) for kpt in eelectron.bands.kpts]) ) ) )
                        print 'in band %d, between experiment and backdrop, post-interpolation Cauchy eigenvalue difference is %.5f.\n' %(idx_band, error_interpd[-1])

                print u'smearing should be larger than numerical energy error: sigma[%.4f] > post-interpolation \u03B4E[%.4f]' %(eelectron.grepen.sigma, np.average(error_interpd))

            elif not eelectron.grepen.is_kpoints_mesh and not belectron.grepen.is_kpoints_mesh:

                if not np.array_equal(eelectron.bands.kpts, belectron.bands.kpts):
                    raise shared.CustomError(self.__class__.__name__ + '.compute: compare neither-mesh cases require same kpoints')

                idx_spin = 0
                if eelectron.grepen.nbands == belectron.grepen.nbands:
                    delta = np.std( abs(eelectron.bands.bands[idx_spin] - belectron.bands.bands[idx_spin]).flatten() )
                    print u'<eigenvalue difference std> between self and belectron (removing offset) is %s eV.\n' %delta
                elif abs(eelectron.grepen.nbands - belectron.grepen.nbands) < belectron.grepen.nbands / 5:     # arbitrary
                    print u'eelectron.grepen.nbands %s and belectron.grepen.nbands %s does not match. trying to guess...\n' %(electron.grepen.nbands, belectron.grepen.nbands)
                    nbands = min(eelectron.grepen.nbands, belectron.grepen.nbands)
                    delta = []
                    for idx_start_eelectron in range(0, eelectron.grepen.nbands-nbands+1):
                        for idx_start_belectron in range(0, belectron.grepen.nbands-nbands+1):
                            delta.append(np.std(abs(eelectron.bands.bands[idx_spin, idx_start_eelectron:idx_start_eelectron + nbands, :] - belectron.bands.bands[idx_spin, idx_start_belectron:idx_start_belectron + nbands, :])))
                    print u'<eigenvalue difference std> between eelectron and belectron (removing offset, guessed) is %s eV.\n' %min(delta)

            elif not eelectron.grepen.is_kpoints_mesh and belectron.grepen.is_kpoints_mesh:

                raise shared.CustomError(self.__class__.__name__ + 'WARNING: non-mesh vs mesh feature has not been implemented!')

        # compare=optimized_cell

        if self.gen.parse_if('etype=cell | btype=cell | etype=ocell | btype=ocell'):

            print '-' * 130

            if not bnode and self.gen.parse_if('etype=ocell & btype=cell'):     # single-node case
                eoc = enode.vasp.optimized_cell
                boc = enode.cell
            else:
                eoc = enode.vasp.optimized_cell if self.gen.parse_if('etype=ocell') else enode.cell
                boc = bnode.vasp.optimized_cell if self.gen.parse_if('btype=ocell') else bnode.cell

            if not np.array_equal(eoc.stoichiometry, boc.stoichiometry):
                raise shared.CustomError(self.__class__.__name__ + '.compute: cell stoichiometry are not the same, cannot compute')

            # base
            print u'<base difference> between self and backdrop is %s A.' % ( np.average( abs(eoc.base - boc.base).flatten() ) )
            print '-' * 130

            # simple difference, not allowing translation or rotation
            print u'<simple cartesian difference> (no translation or rotation allowed) between self and backdrop is  %s A.' % ( np.abs(eoc.ccoor - boc.ccoor).mean() )
            print '-' * 130

            # unordered bijective differences
            compare_cell_bijective(eoc, boc)

            # are they the same cell?
            for ZERO in [0.01, 0.05, 0.1, 0.3]:
                print compare_cell(eoc, boc, ZERO=ZERO)



# =============

# Md


class Md(Dummy):

    @shared.log_wrap
    def compute(self):

        prev = Map().rlookup(attr_dict={'md':self}, prev=True)

        if self.gen.parse_if('movie'):

            # parse vasprun.xml and establish a 'data' nparray, to be used for movie-making
            print "parsing vasprun.xml for trajectory...\n"
            os.chdir(prev.path)
            tree = ET.parse('vasprun.xml')
            root = tree.getroot()
            data = np.zeros(( prev.cell.natoms(), 3, len(root.findall('calculation')) ))
            # each step
            for idx_step, ionicstep in enumerate(root.findall('calculation')):
                structure = ionicstep.find('structure')
                # base
                base = []
                basis = structure.find('crystal').find("varray[@name='basis']")
                for a in basis.findall('v'):
                    base.append(a.text.split())
                base = np.float_(base)
                # fcoor
                fcoor = []
                positions = structure.find("varray[@name='positions']")
                for x in positions.findall('v'):
                    fcoor.append(x.text.split())
                fcoor = np.float_(fcoor)
                # ccoor
                ccoor = np.dot(fcoor, base)
                for idx_traj, c in enumerate(ccoor):
                    data[idx_traj, :, idx_step] = c[:]

            # the user might want to compare initial and final status. for example, total shift, or bad
            # starting lattice parameter.
            # in this case, data gets interpolated
            if self.gen.parse_if('linear=true'):
                for idx_step in range(data.shape[2]):
                    data[:,:,idx_step] = data[:,:,0] + (data[:,:,-1] - data[:,:,0]) / data.shape[2] * idx_step


            """
            Simple 3D animation. https://matplotlib.org/examples/animation/simple_3danim.html
            Data structure of data is data [idx_traj] [idx_dim] [idx_step]
            """
            print "creating movie from trajectory...\n"

            def update_lines(num, dataLines, lines):
                for line, data in zip(lines, dataLines):
                    # NOTE: there is no .set_data() for 3 dim data...
                    line.set_data(data[0:2, :num])
                    line.set_3d_properties(data[2, :num])
                return lines

            # Attaching 3D axis to the figure
            fig = plt.figure()
            ax = p3.Axes3D(fig)
            ax.view_init(float(self.gen.getkw('elev')), float(self.gen.getkw('angle')))

            # This used to be how data is generated
            # data = [Gen_RandLine(25, 3) for index in range(50)]

            # Creating fifty line objects.
            # NOTE: Can't pass empty arrays into 3d version of plot()
            lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

            # Setting the axes properties
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(''.join(k+str(v) for k,v in prev.cell.stoichiometry.iteritems()))

            # Creating the Animation object
            line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
                                               interval=50, blit=False)

            # plt.show()
            print 'movied saved at   %s    .' %prev.path
            line_ani.save('traj_%s.mp4' %(re.sub(r'(\W|\s)+', '', prev.name)), fps=5, writer="avconv", codec="libx264")
