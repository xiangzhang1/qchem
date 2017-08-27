#!/usr/bin/python
# -*- coding: utf-8 -*-

# Gen
# ===========================================================================

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

from itertools import groupby

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

# import progressbar
# from progressbar import Bar, Counter, ETA,FormatLabel, Percentage,ProgressBar
from tqdm import tqdm, trange

from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy import spatial

import shared


class Gen(object):  # Stores the logical structure of keywords and modules. A unique construct deserving a name.

    # Utilities
    # ---------
    def getkw(self, kwname):
        if kwname not in self.kw:
            raise shared.DeferError(self.__class__.__name__ + '. getkw: keyword {%s} not found, DeferError raised' %kwname)
        if len(self.kw[kwname]) != 1:
            raise shared.DeferError(self.__class__.__name__ + '.getkw: self.kw[kwname] does not have 1 unique value, DeferError raised')
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
                #:debug
                if shared.DEBUG:
                    print self.__class__.__name__ + ' parse_require: gave kw {%s} value {%s}' % (kwname, result)
                #;
            if run and not bool(result):
                raise shared.CustomError(self.__class__.__name__ + ' parse_require run=True error: parse_require results in empty set: kwname {%s}, value {%s}, required value {%s}' % (kwname, self.kw[kwname] if kwname in self.kw else 'null', kwvalset))
            if not run and not bool(result) and shared.DEBUG:
                print self.__class__.__name__ + ' parse_require warning: parse_require results in empty set, deferred: kwname {%s}, value {%s}, required_value {%s}' %(kwname, self.kw[kwname] if kwname in self.kw else 'null', kwvalset)
            if self.moonphase > 0:  self.kw_legal_set.add(kwname)
            return bool(result)
        elif 'internal' in expression:      ## parse kwname internal
            kwname = re.sub('internal', '', expression).strip()
            if self.moonphase > 0:  self.kw_internal_set.add(kwname)
            return True
        elif not '(' in expression and not 'null' in expression:    ## parse !modname
            modname = re.sub('!', '', expression).strip()
            if '!' in expression:
                result = (self.mod[modname] if modname in self.mod else set()) - set([True])
            else:
                result = (self.mod[modname] if modname in self.mod else set()) | set([True])
            if run and bool(result):        ### output
                self.mod[modname] = result
            if self.moonphase > 0:  self.mod_legal_set.add(modname)
            return bool(result)
        else:                               ## parse if expression
            result = self.parse_if(expression)
            #:debug
            if not run and not result and shared.DEBUG:
                    print self.__class__.__name__ + ' parse_require warning: parse_require results in empty set, deferred: expression {%s}' %(expression)
            #;
            return result

    def parse_if(self,expression):  # recursively evaluate complex if condition. accepts empty expression.
        #:grammar check
        if ',' in expression:
            raise shared.CustomError( self.__class__.__name__ + ' parse_if error: "," in if expression {%s} in engine.gen.*.conf. Did you mean to use "&"?' %expression)
        #;
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
            if self.moonphase==2 or (self.moonphase==1 and result):  self.kw_legal_set.add(name)
            return result
        elif expression.startswith('(') and expression.endswith(')'):    ## parse (funcname)
            return self.evaluate(expression)
        elif expression == '':                  ## parse empty expression
            return True
        elif 'null' in expression:              ## parse kwname null
            kwname = re.sub('null', '', expression).strip()
            result = not (kwname in self.kw and bool(self.kw[kwname]) )
            if self.moonphase==2 or (self.moonphase==1 and result):  self.kw_legal_set.add(kwname)
            return result
        else:                                   ## parse modname
            if self.moonphase>0:    self.mod_legal_set.add(expression)
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
            outfile.write('KPOINTS\n')
            outfile.write('0\n')
            outfile.write(self.getkw('kpoints').split()[0] + '\n')
            outfile.write( ' '.join(self.getkw('kpoints').split()[1:]) + '\n' )
            outfile.write('0 0 0')

    def pot(self, symbol):
        if len(shared.ELEMENTS[symbol].pot) == 0:
            raise shared.CustomError(' pot: POTCAR for '+symbol+' not found.')
        path = shared.SCRIPT_DIR + '/resource/paw_pbe/'+shared.ELEMENTS[symbol].pot + '/POTCAR'
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


    # main
    # -------------
    def __init__(self, node):
        self.cell = node.cell
        input_ = node.phase + ', ' + node.property
    # 读mod, kw
        self.mod = {}
        self.kw = {}
        self.kw_legal_set = set()
        self.kw_internal_set = set()
        self.mod_legal_set = set()
        self.moonphase = 0
        input_ = [p.strip() for p in input_.split(',') if p.rstrip()]
        for item in input_:
            self.parse_require(item,True)
    # 执行require
        self.require = []
        self.moonphase = 1
        if not [x for x in input_ if x.startswith('engine')]:
            raise shared.CustomError(self.__class__.__name__+': __init__: no engine=x found. Input_: {%s}' %input_)
        engine_name = [x for x in input_ if x.startswith('engine')][0].split('=')[1].strip()
        with open(shared.SCRIPT_DIR + '/conf/engine.gen.' + engine_name + '.conf') as conf:
            lines = conf.read().splitlines()
            for line in [ [p.strip() for p in l.split(':')] for l in lines if not l.startswith('#') ]:
                #:grammar check
                if len(line) < 4: raise shared.CustomError('bad conf grammar error: needs 3 colons per line least in {%s}' %line)
                #;
                for part in [p.strip() for p in line[1].split(',') ]:
                    try:
                        if self.parse_if(line[0]) and self.parse_require(part,False):
                            self.moonphase=2 ; self.parse_require(part,True) ; self.moonphase=1
                        else:
                            self.require.append([line[0],part,line[2],line[3]])
                    except shared.DeferError:
                            self.require.append([line[0],part,line[2],line[3]])
        ## round 2+: got a 'no' in first round
        continue_flag = True
        while continue_flag:
            continue_flag = False
            for line in self.require:
                try:
                    if self.parse_if(line[0]) and self.parse_require(line[1],False):
                        self.moonphase=2 ; self.parse_require(line[1],True) ; self.moonphase=1
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
        if self.parse_if('engine=vasp'):
            self.check_memory()

    def check_memory(self):
        # make temporary dir
        tmp_path = shared.SCRIPT_DIR + '/check_memory_tmp'
        if os.path.exists(tmp_path):
            #raise shared.CustomError('Folder {%s} already exists. Usually do not delete folder to avoid confusion.' %path)
            os.system('trash '+tmp_path)
        os.mkdir(tmp_path)
        os.chdir(tmp_path)
        # alter and write
        wayback = []
        multiplier = 1
        if self.parse_if('isym=-1'):
            wayback.append( 'isym=-1' )
            multiplier *= 2
            self.kw['isym'] = ['0']
        if self.parse_if('lsorbit=.TRUE.'):
            wayback.append( 'lsorbit=.TRUE.' )
            multiplier *= 2
            self.kw['lsorbit'] = ['.FALSE.']
        self.write_incar_kpoints()
        with open('POSCAR','w') as f:
            f.write(self.cell.poscar4())
        for symbol in self.cell.stoichiometry.keys():
            self.pot(symbol)
        if 'isym=-1' in wayback:
            self.kw['isym'] = ['-1']
        if 'lsorbit=.TRUE.' in wayback:
            self.kw['lsorbit'] = ['.TRUE.']
        # calculate and read
        output = check_output([shared.SCRIPT_DIR + '/resource/makeparam']).splitlines()
        try:
            self.memory = {}
            self.memory['arraygrid'] = int( next(l for l in output if 'arrays on large grid' in l).split()[7] )
            self.memory['wavefunction']  = int( next(l for l in output if 'sets of wavefunctions' in l).split()[4] )
            self.memory['projector_real']  = abs(int( next(l for l in output if 'projectors in real space' in l).split()[4] ))
            self.memory['projector_reciprocal']  = abs(int( next(l for l in output if 'projectors in reciprocal space' in l).split()[4] ))
        except StopIteration, KeyError:
            print '\n'.join(output)
            raise shared.CustomError(self.__class__.__name__ + 'error: makeparam output illegal. Check POSCAR4 format and memory leak in script dir.')
        # parse and raise error
        if self.parse_if('hse'):
            memory_required = ( (self.memory['projector_real'] + self.memory['projector_reciprocal'])*int(self.getkw('npar')) + 3*self.memory['wavefunction']*int(self.getkw('kpar')) )/1024/1024/1024 + int(self.getkw('nnode'))
        else:
            memory_required = ( (self.memory['projector_real'] + self.memory['projector_reciprocal'])*int(self.getkw('npar')) + self.memory['wavefunction']*int(self.getkw('kpar')) )/1024/1024/1024 + int(self.getkw('nnode'))*3/2
        memory_required *= multiplier
        memory_available = int(self.getkw('nnode')) * int(self.getkw('mem_node'))
        if memory_required > memory_available:
            print self.__class__.__name__ + ' check_memory warning: insufficient memory. Mem required is {' + str(memory_required) + '} GB. Available mem is {' + str(memory_available) + '} GB.'
        else:
            print self.__class__.__name__ + ' check_memory report: Mem required is {' + str(memory_required) + '} GB. Available mem is {' + str(memory_available) + '} GB.'
        # cleanup
        os.chdir(shared.SCRIPT_DIR)
        shutil.rmtree(tmp_path)

    # User-defined (funcname)
    # -----------------------
    def ncore_total(self):
        return str( int(self.getkw('nnode')) * int(self.getkw('ncore_node')) )

    def nbands(self):
        print self.__class__.__name__ + ' warning: nbands may not be that reliable'
        if self.parse_if('spin=ncl'):
            nbands = ( self.cell.nelectrons * 3 / 5 + sum(self.cell.stoichiometry.values()) * 3 / 2 ) * 2
        elif self.parse_if('spin=para'):
            nbands = self.cell.nelectrons * 3 / 5 + sum(self.cell.stoichiometry.values()) * 1 / 2
        elif self.parse_if('spin=afm|spin=fm'):
            nbands = self.cell.nelectrons / 2 + sum(self.cell.stoichiometry.values()) / 2
        else:
            raise shared.CustomError(self.__class__.__name__+'spin variable is not fm, afm or para, cannot compute nbands')
        # hse case when hse mod is not even defined. for ref, i think. hiya, later self.
        if self.parse_if('hse|prehf'):
            npar = int(self.getkw('npar'))
            nbands = (nbands + npar -1 ) / npar * npar
        return str(nbands)

    def lmaxmix(self):
        b_l_map = { 's': 2, 'p': 2, 'd': 4, 'f': 6, 'g': 8 }
        lmaxmix = max( [ b_l_map[ shared.ELEMENTS[symbol].block ] for symbol in  self.cell.stoichiometry.keys() ] )
        return str(lmaxmix)

    def encut(self):
        result = max( [ shared.ELEMENTS[symbol].pot_encut for symbol in self.cell.stoichiometry.keys() ] )
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
        # NEED band support
        if kpoints[0] not in 'GM':
            raise shared.CustomError(self.__class__.__name__ + '.ismear5check: First member of kpoints, {%s}, is not supported (need band support)' %kpoints[0])
        return np.prod([int(x) for x in kpoints[1:] ]) > 2

    def kpointscheck(self):
        '''kpoints format is sane'''
        kpoints = self.getkw('kpoints').split()
        # NEED band support
        if kpoints[0] in 'GM' and len(kpoints)==4:
            return True
        else:
            raise shared.CustomError(self.__class__.__name__ + '.kpointscheck: Kpoints format wrong. ')
        if kpoints[0] != 'G':
            print self.__class__.__name__ + '.kpointscheck warning: In general, for low-symmetry cells it is sometimes difficult to symmetrize the k-mesh if it is not centered on Gamma. For hexagonal cell, it becomes indeed impossible.'
        return True

    def nkred_divide(self):
        kpoints = self.getkw('kpoints').split(' ')
        return kpoints[0] in 'GM'\
               and int(kpoints[1]) % int(self.getkw('nkredx')) == 0 \
               and int(kpoints[2]) % int(self.getkw('nkredy')) == 0 \
               and int(kpoints[3]) % int(self.getkw('nkredz')) == 0

    def magmom(self):
        magmom = ''
        if self.parse_if('spin=afm'):
            print self.__class__.__name__ + ' warning: more than 1 AFM pattern exists.'
            for symbol in self.cell.stoichiometry:
                l = [0] * self.cell.stoichiometry[symbol]
                base = shared.ELEMENTS[symbol].magmom
                l[::2] = base
                l[1::2] = -1 * base
                magmom += ' ' + ' '.join(l)
        if self.parse_if('spin=fm'):
            for symbol in self.cell.stoichiometry:
                base = shared.ELEMENTS[symbol].magmom
                magmom += ' ' + str( self.cell.stoichiometry[symbol] ) + '*' + str( base )
        return magmom
    def ldauu(self):
        ldauu = ''
        for symbol in self.cell.stoichiometry:
            ldauu += str( shared.ELEMENTS[symbol].ldauu )
        return ldauu
    def ldauj(self):
        ldauj = ''
        for symbol in self.cell.stoichiometry:
            ldauj += str( shared.ELEMENTS[symbol].ldauj )
        return ldauj

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
        if not lines[7].startswith('D'):
            raise shared.CustomError(self.__class__.__name__+'__init__: unsupported POSCAR5 format. Only direct coordinates are supported.')
        self.coordinates = np.float32([ line.split() for line in lines[8:8+sum(self.stoichiometry.values())] ])
        for coor in self.coordinates:
            if len(coor)!=3:
                raise shared.CustomError(self.__class__.__name__+'__init__: bad format. Coordinate line {%s}' %coor)
        # some computation
        self.nelectrons = sum( [self.stoichiometry[symbol] * shared.ELEMENTS[symbol].pot_zval for symbol in self.stoichiometry] )

    def __str__(self):
        result = self.name+'\n'
        result += '1\n'
        for line in self.base:
            result += '\t'.join(map(str,line)) + '\n'
        result += '  '.join(self.stoichiometry.keys()) + '\n'
        result += '  '.join(map(str,self.stoichiometry.values())) + '\n'
        result += 'Direct\n'
        for line in self.coordinates:
            result += ' '.join(map(str,line))+'\n'
        return result

    def poscar4(self):
        result = str(self)
        result = '\n'.join( [x for i,x in enumerate(result.splitlines()) if i!=5] )
        return result


# ARCHAIC
class Poscar(object):
    # The old, broken Poscar.
    # reads poscar, and generates 3*3*3 mirror for all kinds of purposes.
    def __init__(self):
        self.log = ''
        #0.parameters:
        #exclude_dist, exlude_pair, truncate_dist_at, exclude_ele_pair
        #1.get system parameters
        f=open("POSCAR","r")
        self.lines=f.readlines()
        self.cell=[self.lines[i].split() for i in range(2,5)]
        self.base=[self.lines[i].split()[0:3] for i in range(8,len(self.lines))]
        self.elements = self.lines[5].split()
        self.nelements = len(self.elements)
        self.atomcounts = np.int_(self.lines[6].split())
        self.natoms = sum(self.atomcounts)
        if len(self.base[-1])==0:
         print 'poscar.py warning: last line of POSCAR should not be empty, watch it! Removing the last line...'
         self.base.pop(-1)
        if len(self.base[-1])==0:
         print 'poscar.py error: last line of POSCAR still empty! '
         exit(-1)
        if any(len(x.strip())==0 for x in self.lines):
            print  'poscar.py error: no empty lines allowed in POSCAR. that is , second half of poscar is not allowed.'
            exit(-1)
        self.cell=np.float64(self.cell)
        self.base=np.float64(self.base)
        #2.image to supercell
        self.pos_imaged=[]
        self.rpos_imaged=[]
        for i in [0,1,-1]:
         for j in [0,1,-1]:
          for k in [0,1,-1]:
           tmp_image_shift=np.float64([i,j,k])
           for id_base in range(len(self.base)):
            ele_pos_imaged=tmp_image_shift+self.base[id_base]
            self.rpos_imaged.append(ele_pos_imaged)
            ele_pos_imaged=np.dot(ele_pos_imaged,self.cell)
            self.pos_imaged.append(ele_pos_imaged)
        self.pos_imaged=np.float64(self.pos_imaged)
        self.rpos_imaged=np.float64(self.rpos_imaged)
        self.pos_original=np.dot(self.base,self.cell)
        self.pos_original=np.float64(self.pos_original)
        #3.calculate distances
        #calculate dist_qui. this is the quintuplet [id1_pos_imaged, id2_pos_imaged, id1_base, id2_base, tmp_dist]
        self.dist_qui=[]
        changefromto=[]
        for id1_base in range(len(self.base)):
         for id2_base in range(len(self.base)):
          for id2_pos_imaged in [id2_base+tmp_image_shift*len(self.base) for tmp_image_shift in range(27)]:
           tmp_dist=np.linalg.norm(self.pos_original[id1_base]-self.pos_imaged[id2_pos_imaged])
           if abs(tmp_dist)<0.1:
            continue
           # add 0.4 to distances when necessary, to separate say Pb from S.
           # i is the index of first atom, starting from 0. j is that of the second. k is the index of imaged second atom.
           #if i>1 and j>1:
            #dist=dis`t+0.4
           id1_pos_imaged=id1_base
           self.dist_qui.append([id1_pos_imaged,id2_pos_imaged,id1_base,id2_base,tmp_dist])
        self.dist_qui=np.float64(self.dist_qui)
        self.dist_qui=self.dist_qui[np.argsort(self.dist_qui[:,4])]
        #get the excluded distances using exclude_ele_pair
        self.exclude_pairs = []
        self.exclude_dists = []
        for ele_exclude_pair in self.exclude_pairs:
            ele_exclude_pair=np.float64(ele_exclude_pair)
            exclude_quis=[ele_dist_qui for ele_dist_qui in self.dist_qui if ele_dist_qui[2]==ele_exclude_pair[0] and ele_dist_qui[3]==ele_exclude_pair[1]]
            if exclude_quis != []:
                self.exclude_dists.append(exclude_quis[0][4])
        self.exclude_dists = np.float64(self.exclude_dists)

# ===========================================================================


class Map(object):

    def rlookup(self, attr_list={}, node_list=[], parent=False, unique=True):
        l = self.lookup('master').map.traverse()
        # find the node specifiied by attr_list OR node_list
        children = set()
        for n in l:
            if node_list and n in node_list:
                children.add(n)
            if attr_list and all( [getattr(n,key,None)==attr_list[key] for key in attr_list] ):
                children.add(n)
        # find their common parents
        parents = set()
        for n in l:
            if getattr(n,'map',None) and all( [x in n.map for x in children] ):
                parents.add(n)
        # post-process
        result = parents if parent else children
        if unique and len(result)>1:
            raise shared.CustomError('RLookup: result is not unique. Criterion is: attr_list:{%s} node_list:{%s}' %(attr_list, [x.name for x in node_list]))
        return next(iter(result)) if unique else result


    def lookup(self, name):
        if name == 'master':
            if name in shared.NODES:   return shared.NODES['master']
            else: raise shared.CustomError('找不到master了，求喂食')
        elif name in shared.NODES:
            return shared.NODES.pop(name)
        elif any([x.name == name for x in self._dict]):
            return [x for x in self._dict if x.name == name][0]
        elif '.' in name:
            return self.lookup('.'.join(name.split('.')[:-1])).map.lookup(name.split('.')[-1])
        else:
            raise shared.CustomError(self.__class__.__name__ + ' lookup: Node %s not found' %name)

    def prev(self, node):
        l = [x for x in self._dict if node in self._dict[x]]
        if len(l) > 1:
            raise shared.CustomError(self.__class__.__name__ + ' prev: %s has more than 1 prev node. (wtf?)' %name)
        elif len(l) == 1:
            return l[0]
        else:
            return None

    def traverse(self):
        result = set([x for x in self])
        for n in [x for x in self._dict if getattr(x,'map',None)]:
            result = result | set( n.map.traverse() )
        return result

    def __init__(self, text=''):
        #:initialize dict and text
        self._dict, self._dict2 = {}, {}
        text = text.split('\n')
        #;

        # src -> dst
        for line in text:
            if not line.rstrip():   continue
            line = [x.strip() for x in re.split('(->|-->)', line)]
            #:lone node case
            if len(line) == 1:
                src = self.lookup(line[0])
                if src not in self._dict:   self._dict[src] = []
            #;
            elif len(line) == 3:
                src, dst = self.lookup(line[0]), self.lookup(line[2])
                #:add src, dst to dict
                if src not in self._dict:
                    self._dict[src] = []
                if dst not in self._dict:
                    self._dict[dst] = []
                #;
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




class Vasp(object):

    def __init__(self, node):
        self.gen = node.gen
        self.cell = node.cell
        self.path = node.path
        self.prev = node.prev


    def compute(self):

        #:debug msg
        if shared.DEBUG==2:    print 'calling %s(%s).compute' %(self.__class__.__name__, getattr(self,'path',''))
        #;

        if not getattr(self, 'wrapper', None):
            if os.path.exists(self.path):
                raise shared.CustomError( self.__class__.__name__ + ' __init__: path {%s} already exists. enforcing strictly, you need to remove it manually.' %self.path )
            os.makedirs(self.path)
            os.chdir(self.path)
            if self.gen.parse_if('icharg=1|icharg=11'):
                shutil.copyfile(self.prev.path+'/CHGCAR', self.path+'/CHGCAR')
            if self.gen.parse_if('icharg=0|icharg=10|istart=1|istart=2'):
                shutil.copyfile(self.prev.path+'/WAVECAR', self.path+'/WAVECAR')
            if getattr(self, 'prev', None) and getattr(self.prev, 'gen', None) and self.prev.gen.parse_if('opt') and os.path.isfile(self.prev.path+'CONTCAR'):
                shutil.copyfile('CONTCAR','POSCAR')
            # write incar etc. Relies on inheritance.
            os.chdir(self.path)
            self.gen.write_incar_kpoints()
            with open('POSCAR','w') as f:
                f.write(str(self.cell))
            for symbol in self.cell.stoichiometry.keys():
                self.pot(symbol)
            # setting variables
            ncore_total = str(  int(self.gen.getkw('nnode')) * int(self.gen.getkw('ncore_node'))  )
            if self.gen.parse_if('spin=ncl'):   # vasp flavor
                flavor = 'ncl'
            elif all([int(x)==1 for x in self.gen.getkw('kpoints').split()[0:3]]):
                flavor = 'gam'
            elif self.gen.getkw('nnode') == '0':
                ncore_total = 1
                flavor = 'gpu'
                print self.__class__.__name__ + ': vasp_gpu'
            else:
                flavor = 'std'
            self.remote_folder_name = self.path.split('/')[-2] + '_' + self.path.split('/')[-1] + '_' + hashlib.md5(self.path).hexdigest()[:5] + '_' + str(time.time())
            # write scripts and instructions
            # subfile actually runs vasp. wrapper submits the subfile to system.
            self.wrapper = '#!/bin/bash\n' ; self.subfile = '#!/bin/bash\n'
            if self.gen.parse_if('platform=dellpc'):
                self.subfile += 'echo $PWD `date` start; echo '+'-'*75+'\n'
                self.subfile += 'mpiexec.hydra -n %s /home/xzhang1/src/vasp.5.4.1/bin/vasp_%s </dev/null \n' %(ncore_total, flavor)
                self.subfile += 'mail -s "VASP job finished: {${PWD##*/}}" 8576361405@vtext.com <<<EOM \n'
                self.subfile += 'echo $PWD `date` end  ; echo '+'-'*75+'\n'
                self.wrapper += 'nohup ./subfile 2>&1 >> run.log &'
            if self.gen.parse_if('platform=nanaimo'):
                self.wrapper += 'rsync -a . nanaimo:~/%s\n' %self.remote_folder_name
                self.wrapper += 'ssh nanaimo <<EOF\n'
                self.wrapper += ' cd %s\n' %self.remote_folder_name
                self.wrapper += ' sbatch --nodes=%s --ntasks=%s --job-name=%s -t 12:00:00 --export=ALL subfile\n' %(self.gen.getkw('nnode'), ncore_total, self.remote_folder_name)
                self.wrapper += 'EOF\n'
                self.subfile += '#!/bin/bash\n. /usr/share/Modules/init/bash\nmodule purge\nmodule load intel\nmodule load impi\nmpirun -np %s /opt/vasp.5.4.4/bin/vasp_%s' %(ncore_total, flavor)
            if self.gen.parse_if('platform=irmik'):
                self.wrapper += 'rsync -a . irmik:~/%s\n' %self.remote_folder_name
                self.wrapper += 'ssh irmik <<EOF\n'
                self.wrapper += ' cd %s\n' %self.remote_folder_name
                self.wrapper += ' sbatch --nodes=%s --ntasks=%s --job-name=%s -t 12:00:00 --export=ALL subfile\n' %(self.gen.getkw('nnode'), ncore_total, self.remote_folder_name)
                self.wrapper += 'EOF\n'
                self.subfile += '#!/bin/bash\n. /usr/share/Modules/init/bash\nmodule purge\nmodule load mvapich2-2.2/intel\nmpirun -np %s /opt/vasp.5.4.4/bin/vasp_%s' %(ncore_total, flavor)
            with open('wrapper','w') as of_:
                of_.write(self.wrapper)
                os.system('chmod +x wrapper')
            if self.subfile:
                with open('subfile','w') as of_:
                    of_.write(self.subfile)
                os.system('chmod +x subfile')
            # print self.__class__.__name__ + ': %s ready to be computed. Run wrapper or press y.'
            # if raw_input() == 'y':
            #    os.system(wrapper)
            print '-'*50 + '\n' + self.__class__.__name__ + ': wrapper generated at   %s   , waiting for filesystem update. ' %self.path

        elif not getattr(self,'log',None):
            os.chdir(self.path)
            # log
            l = os.listdir(self.path)
            filename = [x for x in l if x.startswith(('slurm-','run.log','OSZICAR'))][0]
            with open(filename,'r') as if_:
                self.log = if_.read()
            # write parent cell if opt
            parent_node = Map().rlookup(attr_list={'vasp':self}, node_list=[self.prev], unique=True, parent=True)
            if self.prev.gen.parse_if('opt'):
                with open('CONTCAR','r') as infile:
                    text = infile.read()
                    setattr(parent_node, 'cell', Cell(text))
                    setattr(self, 'optimized_cell', Cell(text))

        else:
            print self.__class__.__name__ + ' compute: calculation already completed at %s. Why are you here?' %self.path

    @shared.moonphase_wrap
    def moonphase(self):
        #:debug benchmark msg
        if shared.DEBUG==2:    print 'calling %s(%s).moonphase' %(self.__class__.__name__, getattr(self,'path',''))
        #;

        if not getattr(self, 'wrapper', None):
            return 0

        elif not getattr(self, 'log', None):

            # implements the choke mechanism. instead of reporting computable, report choke. unless detects computation complete, then report success/fail
            if not os.path.exists(self.path):
                return -1
                print self.__class__.__name__ + ' moonphase: could not locate the original path {%s}' %(self.path)

            # vasp_is_running
            if (self.gen.parse_if('platform=dellpc')):
                try:
                    pgrep_output = check_output(['pgrep','vasp'])
                    vasp_is_running = pgrep_output.strip() != ''
                except CalledProcessError:
                    vasp_is_running = False
            elif self.gen.parse_if('platform=nanaimo|platform=irmik'):
                #:debug msg
                if shared.DEBUG==2: print self.__class__.__name__ + '.moonphase: asking %s for status of {%s}' %(self.gen.getkw('platform'), self.path)
                #;
                ssh = paramiko.SSHClient()
                #:paramiko config
                ssh._policy = paramiko.WarningPolicy()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh_config = paramiko.SSHConfig()
                user_config_file = os.path.expanduser("~/.ssh/config")
                if os.path.exists(user_config_file):
                    with open(user_config_file) as f:
                        ssh_config.parse(f)
                #;
                ssh.load_system_host_keys()
                ssh.connect(self.gen.getkw('platform'), username='xzhang1')
                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("squeue -n %s" %self.remote_folder_name)
                result = ssh_stdout.read().strip()
                vasp_is_running = ( len(result.splitlines()) > 1 )
            else:
                raise shared.CustomError(self.__class__.__name__ + '.moonphase: i don\'t know what to do')

            # change to vasprun.xml directory
            if self.gen.parse_if('platform=dellpc'):
                os.chdir(self.path)
            elif self.gen.parse_if('platform=nanaimo|platform=irmik'):
                #:check sshfs mounted
                tmp_path = '%s/Shared/%s' % (shared.HOME_DIR, self.gen.getkw('platform'))
                if not os.listdir(tmp_path):
                    raise shared.CustomError(self.__class__.__name__ + '.moonphase: platform %s not mounted using sshfs' %(self.gen.getkw('platform')))
                #;
                tmp_path = '%s/Shared/%s/%s' % (shared.HOME_DIR, self.gen.getkw('platform'), self.remote_folder_name)
                if not os.path.isfile(tmp_path + '/vasprun.xml'):
                    return 1
                else:
                    os.chdir(tmp_path)
            else:
                raise shared.CustomError(self.__class__.__name__ + '.moonphase: i don\'t know what to do')

            # inspect vasprun.xml
            if os.path.isfile('vasprun.xml') and os.path.getmtime('vasprun.xml')>os.path.getmtime('wrapper') and not vasp_is_running :
                with open('vasprun.xml','r') as if_:
                    if if_.read().splitlines()[-1] != '</modeling>' and not os.path.isfile('.moonphase'):
                        print(self.__class__.__name__+'compute FYI: Vasp computation at %s went wrong. vasprun.xml is incomplete. Use .moonphase file to overwrite.' %self.path)
                        return -1
                    else:
                        # download folder
                        if self.gen.parse_if('platform=nanaimo|platform=irmik|platform=hodduk'):
                            print '%s %s.moonphase: copying remote folder {%s} back to self.path {%s} %s' %(shared.bcolors.WARNING, self.__class__.__name__, self.remote_folder_name, self.path, shared.bcolors.ENDC)
                            subprocess.Popen(['rsync', '-a', '-h', '--info=progress2', '%s:%s/' %(self.gen.getkw('platform'),self.remote_folder_name), '%s'%self.path], stdout=sys.stdout, stderr=sys.stderr).wait()
                            #os.system('scp -r /home/xzhang1/Shared/%s/%s/ %s' %(self.gen.getkw('platform'), self.remote_folder_name, self.path))
                            print self.__class__.__name__ + '.moonphase: copy complete.'
                        self.compute()
                        return 2
            else:
                return 1

        else:   # getattr(self,'log',None)!=None
            return 2

    def delete(self):
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)

    def __str__(self):
        #: return log and optimized_cell
        if getattr(self, 'optimized_cell', None):
            return '# optimized_cell:\n' + str(self.optimized_cell) + self.log
        if getattr(self, 'log', None):
            return self.log
        else:
            return 'moonphase is not 2, nothing here'
        #;


    def pot(self, symbol):
        if len(shared.ELEMENTS[symbol].pot) == 0:
            raise shared.CustomError(' pot: POTCAR for '+symbol+' not found.')
        path = shared.SCRIPT_DIR + '/resource/paw_pbe/'+shared.ELEMENTS[symbol].pot + '/POTCAR'
        if_ = open(path,'r')
        of_ = open('./POTCAR','a')
        of_.write( if_.read() )



#===========================================================================

class Dummy(object):

    def __init__(self, node):
        self.path = node.path
        os.mkdir(self.path)

    def compute(self):

        if not os.path.isdir(self.path):
            os.mkdirs(self.path)
        dcmp = dircmp(self.prev.path, self.path)
        if dcmp.left_only or dcmp.right_only or dcmp.diff_files:
            shutil.copytree(self.prev.path, self.path)

    @shared.moonphase_wrap
    def moonphase(self):
        if not os.path.isdir(self.path):
            return 0
        dcmp = dircmp(self.prev.path, self.path)
        if dcmp.left_only or dcmp.right_only or dcmp.diff_files:
            return 0
        return 2

    def delete(self):
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)


#===========================================================================

# Electron


class Grepen(object):

    @shared.log_wrap
    def __init__(self, prev_gen):

        self.energy=float(os.popen('grep "energy without" OUTCAR | tail -1 | awk \'{print $5}\'').read())
        self.efermi=float(os.popen('grep "E-fermi" OUTCAR | awk \'{print $3}\'').read())

        self.nbands=int(os.popen('grep NBANDS OUTCAR | awk \'{print $15}\'').read())
        self.nedos=int(os.popen('grep NEDOS OUTCAR | awk \'{print $6}\'').read())

        self.spin = prev_gen.getkw('spin')
        self.ismear = int(prev_gen.getkw('ismear'))
        self.sigma = 0 if self.ismear!=0 else float(prev_gen.getkw('ismear'))
        self.prev_gen = prev_gen

        with open('DOSCAR','r') as doscar_file:
            self.is_doscar_usable = len(doscar_file.readlines()) > 7

        with open('KPOINTS','r') as kpoints_file:
            kpoints = prev_gen.getkw('kpoints').split()
            self.is_kpoints_mesh = ( kpoints[0] in 'GM' and np.prod(np.float32(kpoints[1:]))>4 and len(kpoints_file.readlines())<7 )

        with open("EIGENVAL","r") as eigenval_file:
            eigenval = [ x.split() for x in eigenval_file.readlines() ]
            self.temperature = float( eigenval[2][0] )
            self.nelectrons = int( eigenval[5][0] )
            self.nkpts = int( eigenval[5][1] )
            if (self.nkpts != len(eigenval) / (self.nbands+2)):
                raise shared.CustomError(self.__class__.__name__ + '__init__: EIGENVAL file length not matching nkpts.')

        for name, value in vars(self).iteritems():
            if name != 'log':
                self.log += '%s %s %s: %s\n' % (shared.bcolors.BOLD, name, shared.bcolors.ENDC, value)


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
                        for idx_atom in range(sum(self.cell.stoichiometry.values())) ] \
                            for idx_spin in range(self.nspins_pdos) ]

    @shared.log_wrap
    def __init__(self, grepen, cell):
        ZERO = 0.001
        self.nspins_dos = {'para':1, 'fm':2, 'ncl':1}[grepen.spin]
        self.nspins_pdos = {'para':1, 'fm':2, 'ncl':3}[grepen.spin]
        self.grepen = grepen
        self.cell = cell
        self.log += 'Fermi level = %s\n\n' % (self.grepen.efermi)
        if not grepen.is_doscar_usable:
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
        self.idx_fermi = abs(self.dos[0, :, 0] - self.grepen.efermi).argmin() + 1
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
            self.log += "spin %s: VBM %s, CBM %s, bandgap %s eV\n" % (idx_spin, self.bandgap[idx_spin][0], self.bandgap[idx_spin][1], self.bandgap[idx_spin][1]-self.bandgap[idx_spin][0]) \
                  if i != j else "spin %s: no bandgap\n" % (idx_spin)

        # pdos
        self.norbitals_pdos = ( len(doscar_site[0][0]) - 1 ) / self.nspins_pdos
        self.pdos = np.zeros([self.nspins_pdos, sum(self.cell.stoichiometry.values()), self.norbitals_pdos, len(doscar_site[0]), 2])
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

    # fit the band for verifying smoothness, and interpolating bandgap
    @shared.MWT(timeout=2592000)
    def bands_interp(self):
        return [ [ Rbf(self.kpts[:,0], self.kpts[:,1], self.kpts[:,2], self.bands[idx_spin, idx_band])
                           for idx_band in range(self.grepen.nbands) ] for idx_spin in range(self.nspins_bands) ]

    @shared.debug_wrap
    @shared.log_wrap
    def __init__(self, grepen):
        ZERO = 0.01
        self.grepen = grepen
        self.nspins_bands = {'para':1, 'fm':2, 'ncl':1}[grepen.spin]

        # bands
        with open("EIGENVAL","r") as f:
            eigenval = [x.split() for x in f.readlines()]
        self.kpts = np.zeros([grepen.nkpts,3])
        self.bands = np.zeros([self.nspins_bands, grepen.nbands, grepen.nkpts])
        #:pop header lines
        del eigenval[:6]
        #;
        for idx_kpt in trange(grepen.nkpts, leave=False, desc='reading bands'):
            #
            #:pop [empty line]
            eigenval.pop(0)
            #;
            eigenval_ = eigenval.pop(0)
            self.kpts[idx_kpt,:] = eigenval_[:3]
            #
            for idx_band in range(grepen.nbands):
                eigenval_ = eigenval.pop(0) ; eigenval_.pop(0)
                for idx_spin in range(self.nspins_bands):
                    self.bands[idx_spin, idx_band, idx_kpt] = eigenval_.pop(0)

        # delta_k
        min_kpt_dist = np.amin( spatial.distance.pdist(self.kpts, metric='Euclidean'), axis=None )   # spatial.distance.pdist() produces a list of distances. amin() produces minimum for flattened input
        kpts_nn = spatial.cKDTree( self.kpts )                                                        # returns a KDTree object, which has interface for querying nearest neighbors of any kpt
        kpts_nn_list = kpts_nn.query_pairs(r=min_kpt_dist*1.5, output_type='ndarray')           # gets all nearest-neighbor idx_kpt pairs
        self.log += u"kpoint mesh precision \u0394k = %.5f 2\u03C0/a.\n" %(min_kpt_dist)
        self.log += '-' * 70 + '\n'

        # bandgap
        self.bandgaps = [ [] for idx_spin in range(self.nspins_bands) ]
        for idx_spin in range(self.nspins_bands):
            vbm = max([e for e in tqdm(np.nditer(self.bands[idx_spin]), leave=False) if e<=grepen.efermi + ZERO])    # else VB slightly beyond efermi is considered CB
            cbm = min([e for e in tqdm(np.nditer(self.bands[idx_spin]), leave=False) if e>=grepen.efermi + ZERO])    # np.nditer is an iterator looping over all dimensions of an array.
                                                                                               # the array itself is an iterator looping normally by outmost dimension.
            self.bandgaps[idx_spin] = [vbm, cbm] if cbm > vbm + ZERO else []
            self.log += "spin %s: VBM %s at %s, CBM %s at %s, kpoint-independent bandgap %s eV\n" \
                  % (idx_spin, vbm, self.kpts[ np.where(self.bands[idx_spin]==vbm)[0][0] ], cbm, self.kpts[ np.where(self.bands[idx_spin]==cbm)[0][0] ], cbm-vbm) \
                  if cbm > vbm + ZERO else "spin %s: no bandgap\n" % (idx_spin)     # only first instance is printed.
        self.log += '-' * 70 + '\n'

        # neargap bands, delta_e
        # calculate DeltaE_KPOINTS by grabbing average E diff / average E diff near bandgap from EIGENVAL.
        # for each spin
        for idx_spin in range(self.nspins_bands):
            ZERO = 0.01
            if not self.bandgaps[idx_spin]: # conductor
                self.log += u'spin %s: no bandgap, \u3B4E skipped.\n' % idx_spin ; continue
            if [idx2_spin for idx2_spin in range(idx_spin) if self.bandgaps[idx2_spin] and np.linalg.norm(np.subtract(self.bangaps[idx_spin], self.bandgaps[idx2_spin])) < ZERO]:    # repetitive
                self.log += u'spin %s: repetitive, \u3B4E skipped. \n' % ( idx_spin ) ; continue
            # specify neargap criterion ZERO
            self.log += u'spin %s, nearest neighbor \u03B4E:\n' % (idx_spin)
            bandgap = abs(np.subtract(*self.bandgaps[idx_spin]))
            for ZERO in tqdm([bandgap, bandgap/2, bandgap/4], leave=False, desc='calculating delta_e in neargap bands'):
                delta_e_flat = []
                # for each NN pair, compute |delta_e| if energy is within bound
                for idx_band in range(grepen.nbands):
                    for kpts_nn_list_ in kpts_nn_list:  # kpts_nn_list_ = [ idx1_kpt idx2_kpt ]
                        if all(self.bandgaps[idx_spin][0]-ZERO < self.bands[idx_spin][idx_band][idx_kpt] < self.bandgaps[idx_spin][1]+ZERO for idx_kpt in kpts_nn_list_):    # is near gap
                            delta_e_flat.append( abs(self.bands[idx_spin][idx_band][kpts_nn_list_[0]] - self.bands[idx_spin][idx_band][kpts_nn_list_[1]]) )
                self.log += u'  CBM/VBM +- %.2f eV: \u03B4E = %.5f eV, # of kpts = %d.\n' %( ZERO, np.mean(delta_e_flat), len(delta_e_flat) ) \
                            if delta_e_flat else u'  CBM/VBM +- %.2f eV: # of kpts = 0.\n' %( ZERO )

        #: interpolated bandgap
        self.log += 'Usually bandgap is between interpolated and raw bandgap. \n'
        #;
        if grepen.is_kpoints_mesh:
            self.bandgaps_interp = [ [] for idx_spin in range(self.nspins_bands) ]
            # define constraint (see docs)
            convex_hull = scipy.spatial.ConvexHull(self.kpts)
            facets = convex_hull.equations  # [[A,B,C,D], ...]
            delaunay = scipy.spatial.Delaunay(self.kpts)
            def abcroot(facets_):
                return np.linalg.norm(facets_[:3])
            def constraint(kpt, facets=facets, delaunay=delaunay):
                sign = 1 if delaunay.find_simplex(kpt) >= 0 else -1
                min_dist = np.amin( np.divide( np.dot(facets, np.append(kpt,1)), np.apply_along_axis(abcroot, 1, facets) ) ) * sign
                return min_dist + min_kpt_dist
            # optimize for each spin and each band
            for idx_spin in range(self.nspins_bands):
                #:relevant parameters exist
                ZERO = 0.01
                if not self.bandgaps[idx_spin]: # conductor
                    self.log += u'spin %s: no bandgap, bandgaps_interp skipped.\n' % ( idx_spin ) ; continue
                if [idx2_spin for idx2_spin in range(idx_spin) if self.bandgaps[idx2_spin] and np.linalg.norm(np.subtract(self.bangaps[idx_spin], self.bandgaps[idx2_spin])) < ZERO]:    # repetitive
                    self.log += u'spin %s: repetitive, bandgaps_interp skipped.\n' % ( idx_spin ) ; continue
                #;
                kptes = []
                ZERO = abs(np.subtract(*self.bandgaps[idx_spin])) / 2.5
                for idx_band in range(grepen.nbands):
                     #: speedup, and max/min
                    if any(self.bandgaps[idx_spin][0] - ZERO < e < self.bandgaps[idx_spin][1] + ZERO for e in self.bands[idx_spin, idx_band]):
                        for sign in (-1,1):
                            #;
                            result = scipy.optimize.fmin_slsqp(
                                                      lambda x, self=self, sign=sign: sign * self.bands_interp()[idx_spin][idx_band](*x) ,
                                                      x0 = self.kpts[ np.where(self.bands[idx_spin]==self.bandgaps[idx_spin][0])[0][0] ],
                                                      f_ieqcons = constraint,
                                                      tol = 1e-3)
                            kptes.append([result.x, result.fun])
                kpt_at_vbm, vbm = np.amax([kpte for kpte in kptes if self.bandgaps[idx_spin][0]-ZERO<kpte[1]<self.bandgaps[idx_spin][0]+ZERO], axis=1)
                kpt_at_cbm, cbm = np.amin([kpte for kpte in kptes if self.bandgaps[idx_spin][1]-ZERO<kpte[1]<self.bandgaps[idx_spin][1]+ZERO], axis=1)
                # self.bandgap_interp
                self.log += "spin %s, interpolated: VBM %s at %s , CBM %s at %s, bandgap %s eV\n" \
                      % (idx_spin, vbm, kpt_at_vbm, cbm, kpt_at_cbm, cbm-vbm) \
                      if cbm > vbm else "spin %s, interpolated: no bandgap\n" % (idx_spin)
            self.log += '-' * 70 + '\n'
        else:
            self.log += 'kpoints is not mesh, bandgap_interp skipped. \n'

    # plot band: slightly broken
    def plot(self,i):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = self.bands[i][:,0]
        ys = self.bands[i][:,1]
        zs = self.bands[i][:,2]
        cs = self.bands[i][:,3]
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
    def __init__(self, cell, grepen, dos):
        # sanity check
        if dos.norbitals_pdos > len(shared.ELEMENTS.orbitals):
            raise shared.CustomError(self.__class__.__name__ +' __init__: more orbitals than supported.')

        # let's start!
        # pristine electronic configuration
        self.log += '\n\nGS electron configurations for elements in POSCAR\n'   # a good idea is to indent within header
        for element in cell.stoichiometry:
            self.log += "%s: %s" %(element, shared.ELEMENTS[element].eleconfig)

        # integrate pdos scaled
        idx_atom = 0
        for symbol, natoms in cell.stoichiometry.iteritems():
            for idx_atom in range(idx_atom, idx_atom + natoms):
                for idx_spin in range(dos.nspins_pdos):
                    self.log += "%s.%s s%s\t" % ( idx_atom, element, idx_spin )
                    for idx_orbital in range(dos.norbitals_pdos):
                        INFINITY = np.argmax( dos.pdos[idx_spin, idx_atom, idx_orbital, :, 0 ] > dos.efermi+5 )
                        integrated_pdos = np.trapz( dos.pdos[idx_spin, idx_atom, idx_orbital, :dos.idx_fermi, 1 ] , \
                                                    x = dos.pdos[idx_spin, idx_atom, idx_orbital, :dos.idx_fermi, 0 ] ) \
                                          / \
                                          np.trapz( dos.pdos[idx_spin, idx_atom, idx_orbital, :INFINITY, 1 ] , \
                                                                      x = dos.pdos[idx_spin, idx_atom, idx_orbital, :INFINITY, 0 ] )
                        self.log += '%s %.2f' % (shared.ELEMENTS[idx_orbital], integrated_pdos)
                    self.log += '\n'

        # Bader charge
        self.log += "\n\nBader charge. Boundaries are defined as zero-flux surfaces. Note that certain flags should be set (e.g. LAECHG) for this to be reasonable.\n"
        os.popen('bader CHGCAR').read()
        with open('ACF.dat','r') as f:
            lines = f.readlines()
        for idx_element, element in enumerate(cell.stoichiometry.keys()):
            self.log += element + ' '
            for idx_atom in range(sum(cell.stoichiometry.values()[0:idx_element]),sum(cell.stoichiometry.values()[0:idx_element+1])):
                nline = idx_atom + 2
                self.log += lines[nline].split()[4] + ' '

        # OUTCAR RWIGS [decomposed] charge
        self.log += "\n\nTotal charge inside the Wigner-Seitz Radius in OUTCAR\n"
        with open('OUTCAR','r') as f:
            lines = f.readlines()
            for idx_line,line in enumerate(lines):
                if '# of ion' in line: # this is only an anchor. that particular line doesn't matter.
                    break
            for idx2_line in range(idx_line ,idx_line + 4 + sum(cell.stoichiometry.values())):
                self.log += lines[idx2_line]
        self.log += '\n\n'

        # Bader magnetic moment
        if grepen.spin == 'fm' or grepen.spin == 'afm':
            self.log += "\n\nAn oversimplified version of Bader magnetic moment.\n"
            os.popen('chgsplit.sh CHGCAR').read()
            os.popen('bader cf2').read()
            for idx_element, element in enumerate(cell.stoichiometry.keys()):
                self.log += element + ' '
                for idx_atom in range(sum(cell.stoichiometry.values()[0:idx_element]),sum(cell.stoichiometry.values()[0:idx_element+1])):
                    nline = idx_atom + 2
                    with open('ACF.dat','r') as f:
                        lines = f.readlines()
                        self.log += lines[nline].split()[4]
                self.log += '\n'

        # OUTCAR RWIGS magnetic moment
        if grepen.spin == 'fm' or grepen.spin == 'afm':
            self.log += "\n\nMagnetization (x) [total magnetic moment inside the Wigner-Seitz Radius] in OUTCAR\n"
            with open('OUTCAR','r') as f:
                lines = f.readlines()
                for idx_line,line in enumerate(lines):
                    if 'magnetization (x)' in line: # this is only an anchor. that particular line doesn't matter.
                        break
                for idx2_line in range(idx_line + 2,idx_line + 6 + sum(cell.stoichiometry.values())):
                    self.log += lines[idx2_line]



class Errors(object):

    @shared.log_wrap
    def __init__(self, Agrepen, Ados, Abands, Bgrepen=None, Bdos=None, Bbands=None):

        ## source of error : requirement
        self.de = 0
        self.log += u'explanation: requirement for \u03B4\n'
        ## rule
        if Agrepen.ismear == 0:
            self.log += u'gaussian smearing smoothes out irregularities with size sigma: sigma[%.4f] < \u03B4[%.4f]/2' %(Agrepen.sigma,self.de)
            self.de = max(self.de, Agrepen.sigma * 2)
        ## rule
        self.log += u'sparse kpoints grid may miss in-between eigenvalues. E(j)-E(j\')[%.4f] < \u03B4[%.4f]/2' %(Abands.de_kpoints,self.de)
        self.de = max(self.de, Abands.de_kpoints * 2)
        ## rule
        self.log += u'all details between two DOS points are lost. 10/NEDOS[%.4f] < \u03B4[%.4f]/2' %(10.0/Agrepen.nedos,self.de)
        self.de = max(self.de, 10.0/float(Agrepen.nedos) * 2)
        ## rule
        self.log += 'DOS should not be so fine that kpoint mesh coarseness is obvious. 10/NEDOS[%.4f] > DE_KPOINTS[%.4f]' %(10.0/Agrepen.nedos,Abands.de_kpoints)

        # comparing against dirB
        self.de_interpd = []
        if not Bgrepen:
            self.log += self.__class__.__name__ + ': skipped E(k) numerical error check, aka A <-> B_interp. purely numeric error should be smaller than 0.01 eV. '
        else:
            idx_spin = 0
            self.log += self.__class__.__name__ + ': comparing idx_spin=0 only.'    # faciliate comparison between ncl and fm
            if Bbands.bands[0].shape != Abands.bands[0].shape:
                raise shared.CustomError(self.__class__.__name__ + '.__init__: A and B bands are incompatible')
            if not Bbands.bandgaps[0] or not Abands.bandgaps[0]:
                raise shared.CustomError(self.__class__.__name__ + '.__init__: bandgap not found for spin 0')
            ZERO = (Bbands.bandgaps[idx_spin][1] - Bbands.bandgaps[idx_spin][0]) / 3
            for idx_band, band in tqdm(enumerate(Bbands[idx_spin]), leave=False, desc='interpolating Bbands for comparison'):
                if any(Bbands.bandgaps[idx_spin][0] - ZERO < e < Bbands.bandgaps[idx_spin][1] + ZERO for e in self.bands[idx_spin][idx_band]):
                    self.de_interpd.append( np.average( abs( Abands.bands[idx_spin][idx_band] - np.float32( Bbands.bands_interp()[idx_spin][idx_band](*kpt) for kpt in Abands.kpts ) ) ) )
                    self.log += 'in band %d, between dirA and dirB, interpolation plus Cauchy error is %.5f.\n' %(i_band, self.de_interpd[-1])

            ## rule
            self.log += 'smearing should be larger than numerical energy error: sigma[%.4f] > DE_INTERPD[%.4f]' %(Agrepen.sigma, np.average(self.de_interpd))

        self.log += 'errors.py: you should expect an error around %.4f eV in dirA. \n' %(self.de)



class Electron(object):
    '''the big boss'''

    def __init__(self, node):

        self.gen = node.gen
        self.path = node.path
        self.prev = node.prev


    def compute(self):

        if not getattr(self, 'log', None):
            if os.path.isdir(self.path):
                raise shared.CustomError(self.__class__.__name__ + ' compute: self.path {%s} taken' %self.path)
            print 'copying previous folder... ',
            shutil.copytree(self.prev.path, self.path)
            print 'done',     # return to line start, but do not go to next line
            os.chdir(self.path)

            if self.gen.parse_if('cell'):
                with open('POSCAR','r') as infile:
                    self.cell = Cell(infile.read())

            if self.gen.parse_if('grepen'):
                self.grepen = Grepen(self.prev.gen)

            if self.gen.parse_if('dos'):
                self.dos = Dos(self.grepen, self.cell)

            if self.gen.parse_if('bands'):
                self.bands = Bands(self.grepen)

            if self.gen.parse_if('charge'):
                self.charge = Charge(self.cell, self.grepen, self.dos)

            if self.gen.parse_if('errors'):
                if self.gen.getkw('cur'):
                    nodeB = engine.Map().lookup(self.gen.getkw('cur'))
                    self.errors = Errors(self.grepen, self.dos, self.bands, nodeB.electron.grepen, nodeB.electron.dos, nodeB.electron.bands)
                else:
                    self.errors = Errors(self.grepen, self.dos, self.bands)

            self.log = ''
            for name in ['cell', 'grepen', 'dos', 'bands','charge', 'errors']:
                if getattr(self, name, None) and getattr(getattr(self, name),'log', None):
                    self.log += str( getattr(getattr(self, name), 'log') )

        else:
            raise shared.CustomError(self.__class__.__name__ + ' compute: moonphase is 2. why are you here?')

    @shared.moonphase_wrap
    def moonphase(self):
        return 2 if getattr(self, 'log', None) else 0

    def __str__(self):
        #:return log
        if getattr(self, 'log', None):
            return self.log
        else:
            return 'moonphase is not 2, nothing here'
        #;

    def delete(self):
        #:remove folder
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        #;
