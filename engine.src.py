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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import progressbar
from progressbar import Bar, Counter, ETA,FormatLabel, Percentage,ProgressBar 
from scipy.interpolate import Rbf
from scipy.optimize import minimize
from scipy import spatial

import shared

class Gen(object):   # Stores the logical structure of keywords and modules. A unique construct deserving a name.

    # Utilities
    # ---------
    def getkw(self,kwname):
        if kwname not in self.kw:
            return None
	if len(self.kw[kwname])!=1:
            raise shared.CustomError(self.__class__.__name__ + ' getkw error: self.kw[kwname] does not have 1 and only 1 value. wait till that happens and try again.')
        if not isinstance(next(iter(self.kw[kwname])), basestring):
            raise shared.CustomError(self.__class__.__name__ + ' getkw error: value {%s} of kw {%s} is not string' %(next(iter(self.kw[kwname])), kwname) )
        return next(iter(self.kw[kwname]))

    def evaluate(self,expression):                  # Evaluates expression to string: literal or (funcname)
        if expression.startswith('(') and expression.endswith(')'): 
            func = getattr(self, re.sub('\(|\)','',expression).strip())
            if not func:
                raise shared.CustomError( self.__class__.__name__+' error: bad conf file error. Unable to find function {%s}' % re.sub('\(|\)','',expression) )
            return func()
        else:   # literal
            return expression

    def parse_require(self,expression,run=False):   # Executes single require expression. Accepts empty expression as True.
        operation_map = {
                '='  : lambda x,y : x & y,
                '!=' : lambda x,y : x - y,
                }
        expression = expression.strip()
        if re.search('=',expression):           # evaluation        ## parse kwname!=kwval|(funcname)
            l = [ p.strip() for p in re.split('(!=|=)',expression) ] ; kwname = l[0] ; operator = l[1] ; kwval_expression = l[2]
            kwvalset = set()
            for kwval_part in kwval_expression.split('|'):
                kwval_part = kwval_part.strip()
                kwvalset.add(self.evaluate(kwval_part))
            if kwname in self.kw and bool(self.kw[kwname]):
                result = operation_map[operator](self.kw[kwname],kwvalset)
            else:
                result = operation_map[operator](kwvalset,kwvalset)
            if run and bool(result):        
                self.kw[kwname] = result
                '''if shared.DEBUG: print self.__class__.__name__ + ' parse_require: gave kw {%s} value {%s}' %(kwname,result)'''
            if run and not bool(result):
                raise shared.CustomError( self.__class__.__name__ + ' parse_require run=True error: parse_require results in empty set: kwname {%s}, value {%s}, required value {%s}' %(kwname, self.kw[kwname] if kwname in self.kw else 'null', kwvalset) )
            '''if not run and not bool(result) and shared.DEBUG:
                print self.__class__.__name__ + ' parse_require warning: parse_require results in empty set, deferred: kwname {%s}, value {%s}, required_value {%s}' %(kwname, self.kw[kwname] if kwname in self.kw else 'null', kwvalset) '''
            if self.moonphase>0:    self.kw_legal_set.add(kwname)
            return bool(result)
        elif 'internal' in expression:      ## parse kwname internal
            kwname = re.sub('internal','',expression).strip()
            if self.moonphase>0:  self.kw_internal_set.add(kwname)
            return True
        elif not '(' in expression and not 'null' in expression:    ## parse !modname
            modname = re.sub('!','',expression).strip()
            if '!' in expression:
                result = ( self.mod[modname] if modname in self.mod else set() ) - set([True])
            else:
                result = ( self.mod[modname] if modname in self.mod else set() ) | set([True])
            if run and bool(result):        ### output
                self.mod[modname] = result
            if self.moonphase>0:  self.mod_legal_set.add(modname)
            return bool(result)
        else:                               ## parse if expression
            result = self.parse_if(expression)
            '''if not run and not result and shared.DEBUG:
                    print self.__class__.__name__ + ' parse_require warning: parse_require results in empty set, deferred: expression {%s}' %(expression)'''
            return result

    def parse_if(self,expression):  # recursively evaluate complex if condition. accepts empty expression.
        '''if ',' in expression:
            raise shared.CustomError( self.__class__.__name__ + ' parse_if error: "," in if expression {%s} in engine.gen.*.conf. Did you mean to use "&"?' %expression)'''
        operation_map = {
                '&&'  :  lambda x,y : x and y,
                '||'  :  lambda x,y : x or y,
                '&'    :  lambda x,y : x and y,
                '|'    :  lambda x,y : x or y
                }
        expression = expression.strip()
        if re.search('&&|\|\|',expression):
            l = [ x.strip() for x in re.split('(&&|\|\|)',expression,maxsplit=1) ]
            return operation_map[l[1]](self.parse_if(l[0]),self.parse_if(l[2]))
        elif re.search('&|\|',expression):
            l = [ x.strip() for x in re.split('(&|\|)',expression,maxsplit=1) ]
            return operation_map[l[1]](self.parse_if(l[0]),self.parse_if(l[2]))
        elif '!' in expression:
            return not self.parse_if(expression.replace('!',''))
        elif '=' in expression: # evaluation    ## parse kwname=kwval
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
            kwname = re.sub('null','',expression).strip()
            result = not ( kwname in self.kw and bool(self.kw[kwname]) )
            if self.moonphase==2 or (self.moonphase==1 and result):  self.kw_legal_set.add(kwname)
            return result
        else:                                   ## parse modname
            if self.moonphase>0:    self.mod_legal_set.add(expression)
            return (expression in self.mod and self.mod[expression]==set([True]))        

    def write_incar_kpoints(self):
        with open('INCAR','w') as outfile:
            for name in self.kw:
                if name not in self.kw_internal_set:
    	            outfile.write('    '+name.upper()+' = '+str(self.getkw(name))+'\n')
        with open('KPOINTS','w') as outfile:
            outfile.write('KPOINTS\n')
            outfile.write('0\n')
            outfile.write(self.getkw('kpoints').split()[3] + '\n')
            outfile.write( ' '.join(self.getkw('kpoints').split()[0:3]) + '\n' )
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
            '''if not self.getkw(name):
                print 'What the literal fuck. name is {%s} and value is {%s} and name in self.kw is {%s}' %(name, self.getkw(name), name in self.kw)'''
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
	with open(shared.SCRIPT_DIR + '/engine.gen.' + engine_name + '.conf') as conf:
	    lines = conf.read().splitlines()
            for line in [ [p.strip() for p in l.split(':')] for l in lines if not l.startswith('#') ]:
                '''if len(line) < 4: raise shared.CustomError('bad conf grammar error: needs 3 colons per line least in {%s}' %line)'''
                for part in [p.strip() for p in line[1].split(',') ]:
                    if self.parse_if(line[0]) and self.parse_require(part,False):  
                        self.moonphase=2 ; self.parse_require(part,True) ; self.moonphase=1
                    else:
                        self.require.append([line[0],part,line[2],line[3]])
        ## round 2+: got a 'no' in first round
        continue_flag = True
        while continue_flag:
            continue_flag = False
            for line in self.require:
                if self.parse_if(line[0]) and self.parse_require(line[1],False):
                        self.moonphase=2 ; self.parse_require(line[1],True) ; self.moonphase=1
                        continue_flag = True
                        self.require.remove(line)
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
        self.moonphase=3    # parsing and validating of input_ is complete.
        if self.parse_if('engine=vasp'):
            self.check_memory()

    def check_memory(self):
        # make temporary dir
        path = shared.SCRIPT_DIR + '/check_memory'
        if os.path.exists(path):
            #raise shared.CustomError('Folder {%s} already exists. Usually do not delete folder to avoid confusion.' %path)
            os.system('trash '+path)
        os.mkdir(path)
        os.chdir(path)
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
        os.chdir('..')
        shutil.rmtree('check_memory')

    # User-defined (funcname)
    # -----------------------
    def ncore_total(self):
        return str( int(self.getkw('nnode')) * int(self.getkw('ncore_node')) )

    def nbands(self):
        print self.__class__.__name__ + ' warning: nbands may not be that reliable'
        if self.parse_if('spin=ncl'):
            nbands = ( self.cell.nelect * 3 / 5 + self.cell.nion * 3 / 2 ) * 2
        elif self.parse_if('spin=para'):
            nbands = self.cell.nelect * 3 / 5 + self.cell.nion * 1 / 2
        elif self.parse_if('spin=afm|spin=fm'):
            nbands = self.cell.nelect / 2 + self.cell.nion / 2
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
        if 'kpoints' not in self.kw:
            return False
	kpoints = self.getkw('kpoints').split(' ')
        return np.prod([int(x) for x in kpoints[:-2] ]) > 2

    def kpointscheck(self):
        kpoints = self.getkw('kpoints')
        if not kpoints:
            return False
        if len(kpoints.split()) != 4 or not kpoints.split()[3].startswith(('G','g','M','m')):
            raise shared.CustomError( self.__class__.__name__ + ' error: bad kpoints format. kpoints should be "kx ky kz monkhorst|G"' )
            return False
        if  not kpoints.split()[3].startswith(('G','g')):
            print self.__class__.__name__ + ' kpointscheck warning: In general, for low-symmetry cells it is sometimes difficult to symmetrize the k-mesh if it is not centered on Gamma. For hexagonal cell, it becomes indeed impossible.'
        return True

    def nkred_divide(self):
	try:
	    kpoints = [int(x) for x in self.getkw('kpoints').split(' ')]
	    nkredx = int(self.getkw('nkredx'))
	    nkredy = int(self.getkw('nkredy'))
	    nkredz = int(self.getkw('nkredz'))
	    return kpoints[0]%nkredx==0 and kpoints[1]%nkredy==0 and kpoints[2]%nkredz==0
	except (KeyError, AttributeError, ValueError) as KwError:
            raise shared.CustomError( self.__class__.__name__ + ' error: kx should be divisible by nkredx, etc.' )
	    return False

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
        self.base = np.float_([ line.split() for line in lines[2:5] ]) * float(lines[1])
        self.stoichiometry = OrderedDict( zip(lines[5].split(), [int(x) for x in lines[6].split()]) )
        if not lines[7].startswith('D'):
            raise shared.CustomError(self.__class__.__name__+'__init__: unsupported POSCAR5 format. Only direct coordinates are supported.')
        self.coordinates = np.float_([ line.split() for line in lines[8:8+sum(self.stoichiometry.values())] ])
        for coor in self.coordinates:
            if len(coor)!=3:
                raise shared.CustomError(self.__class__.__name__+'__init__: bad format. Coordinate line {%s}' %coor)
        # some computation
        self.nion = sum(self.stoichiometry.values())
        self.nelect = sum( [self.stoichiometry[symbol] * shared.ELEMENTS[symbol].pot_zval for symbol in self.stoichiometry] )

    '''def __str__(self):
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
    '''
            

# ARCHAIC
# The old, broken Poscar. 
# reads poscar, and generates 3*3*3 mirror for all kinds of purposes.
class Poscar(object):
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
           for id_base in range(0,len(self.base)):
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
        for id1_base in range(0,len(self.base)):
         for id2_base in range(0,len(self.base)):
          for id2_pos_imaged in [id2_base+tmp_image_shift*len(self.base) for tmp_image_shift in range(0,27)]:
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
        '''if name == 'master':
            if name in shared.NODES:   return shared.NODES['master']
            else: raise shared.CustomError('找不到master了，求喂食')
        elif name in shared.NODES:
            return shared.NODES.pop(name)
        elif any([x.name == name for x in self._dict]):
            return [x for x in self._dict if x.name == name][0]
        elif '.' in name:
            return self.lookup('.'.join(name.split('.')[:-1])).map.lookup(name.split('.')[-1])
        else:
            raise shared.CustomError(self.__class__.__name__ + ' lookup: Node %s not found' %name)'''

    def prev(self, node):
        '''l = [x for x in self._dict if node in self._dict[x]]
        if len(l) > 1: 
            raise shared.CustomError(self.__class__.__name__ + ' prev: %s has more than 1 prev node. (wtf?)' %name)
        elif len(l) == 1:
            return l[0]
        else:
            return None'''

    def traverse(self):
        result = set([x for x in self]) 
        for n in [x for x in self._dict if getattr(x,'map',None)]:
            result = result | set( n.map.traverse() )
        return result


    def __init__(self, text=''):
    
        '''self._dict, self._dict2 = {}, {}
        text = text.split('\n')'''

        # src -> dst
        for line in text:
            if not line.rstrip():   continue
            '''line = [x.strip() for x in re.split('(->|-->)', line)]'''
            if len(line) == 1:
                src = self.lookup(line[0])
                if src not in self._dict:   self._dict[src] = []
            elif len(line) == 3:
                src, dst = self.lookup(line[0]), self.lookup(line[2])
                if src not in self._dict:   
                    self._dict[src] = []
                if dst not in self._dict:   
                    self._dict[dst] = []
                '''m = self._dict if line[1]=='->' else self._dict2
                m[src] = [dst] if src not in m else m[src]+[dst]'''
            else:
                raise shared.CustomError(self.__class__.__name__ + '__init__: src -> dst. 3 parts needed')


    def add_node(self, node):    
        # inherit is done on compute
        # same name / same node exceptions are not allowed.
        # we're moving references around, so renaming is bad. instead, use 'duplicate' command intead.
        '''if any([x.name==node.name for x in self._dict]):
            raise shared.CustomError(self.__class__.__name__+' add_node: node with name {%s} already in self._dict. We\'re moving references around, so auto-renaming is bad. Use duplicate if only input is needed.' %node.name)
        else:'''
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
        '''src = self.lookup(src_name)
        dst = self.lookup(dst_name)
        if src in self._dict[dst] or (dst in self._dict2 and src in self._dict2[dst]):
            raise shared.CustomError(self.__class__.__name__ + ' add_edge: dst %s -> src %s link exists' %(dst_name, src_name))
        if dst in self._dict[src]:
            self._dict[src].remove(dst)
            self._dict2[src] = self._dict2[src]+[dst] if src in self._dict2 else [dst]
        elif src in self._dict2 and dst in self._dict2[src]:
            self._dict2[src].remove(dst)
            self._dict[src] = self._dict[src]+[dst] if src in self._dict else [dst]
        else:'''
            self._dict[src] += [dst]

    def del_edge(self, src_name, dst_name):
        '''src = self.lookup(src_name)
        dst = self.lookup(dst_name)
        if src in self._dict[dst] or dst in self._dict2 and src in self._dict2[dst]:
            raise shared.CustomError(self.__class__.__name__ + ' del_edge: dst %s -> src %s link exists' %(dst_name, src_name))
        if dst in self._dict[src]:
            self._dict[src].remove(dst)
        elif dst in self._dict2[src]:
            self._dict2[src].remove(dst)
        else:
            raise shared.CustomError(self.__class__.__name__ + ' del_edge: no edge to delete')'''

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
            foldername = self.path.split('/')[-2] + '_' + self.path.split('/')[-1] + '_' + hashlib.md5(self.path).hexdigest()
            # write scripts and instructions
            # subfile actually runs vasp. wrapper submits the subfile to system.
            self.wrapper = '#!/bin/bash\n' ; self.subfile = '#!/bin/bash\n'
            if self.gen.getkw('platform') == 'dellpc':
                self.subfile += 'echo $PWD `date` start; echo '+'-'*75+'\n'
                self.subfile += 'mpiexec.hydra -n %s /home/xzhang1/src/vasp.5.4.1/bin/vasp_%s </dev/null \n' %(ncore_total, flavor)
                self.subfile += 'mail -s "VASP job finished: {${PWD##*/}}" 8576361405@vtext.com <<<EOM \n'
                self.subfile += 'echo $PWD `date` end  ; echo '+'-'*75+'\n'
                self.wrapper += 'nohup ./subfile 2>&1 >> run.log &'
            if self.gen.getkw('platform') == 'nanaimo':
                self.wrapper += 'rsync -a . nanaimo:~/%s\n' %foldername
                self.wrapper += 'ssh nanaimo <<EOF\n'
                self.wrapper += ' cd %s\n' %foldername
                self.wrapper += ' sbatch --nodes=%s --ntasks=%s --job-name=%s -t 12:00:00 --export=ALL subfile\n' %(self.gen.getkw('nnode'), ncore_total, self.path)
                self.wrapper += 'EOF\n'
                self.subfile += '#!/bin/bash\n. /usr/share/Modules/init/bash\nmodule purge\nmodule load intel\nmodule load impi\nmpirun -np %s /opt/vasp.5.4.4/bin/vasp_%s' %(ncore_total, flavor)
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
            # moonphase=1. should not be called unless moonphase() decides to report success and further compute.
            # write log
            l = os.listdir(self.path)
            filename = [x for x in l if x.startswith(('slurm-','run.log','OSZICAR'))][0]
            with open(filename,'r') as if_:
                self.log = if_.read()
            # write parent cell if opt
            parent_node = Map().rlookup(attr_list={'vasp':self}, node_list=[self.prev], unique=True, parent=True)
            with open('CONTCAR','r') as infile:
                text = infile.read()
                setattr(parent_node, 'cell', Cell(text))
                setattr(self, 'optimized_cell', Cell(text))

        else:
            print self.__class__.__name__ + ' compute: calculation already completed at %s. Why are you here?' %self.path

    @shared.moonphase_wrap
    def moonphase(self):
        if not getattr(self, 'wrapper', None):
            return 0
        elif not getattr(self, 'log', None):    
            # implements the choke mechanism. instead of reporting computable, report choke. unless detects computation complete, then report success/fail
            if not os.path.exists(self.path):
                return -1
                print self.__class__.__name__ + ' moonphase: FYI status is -1 because path doesnt exist'
            os.chdir(self.path)
            try:
                pgrep_output = check_output(['pgrep','vasp'])
                vasp_is_running = pgrep_output.strip() != ''
            except CalledProcessError:
                vasp_is_running = False
            if os.path.isfile('vasprun.xml') and os.path.getmtime('vasprun.xml')>os.path.getmtime('wrapper') and (self.gen.parse_if('platform!=dellpc') or not vasp_is_running) :
                with open('vasprun.xml','r') as if_:
                    if if_.read().splitlines()[-1] != '</modeling>' and not os.path.isfile('.moonphase'):
                        #print(self.__class__.__name__+'compute FYI: Vasp computation at %s went wrong, status code -1. Use .moonphase file to overwrite.' %self.path)
                        return -1
                    else:
                        self.compute()
                        return 2
            else:
                return 1
        else:
            return 2

    def delete(self):
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)

    '''def __str__(self):
        if getattr(self, 'optimized_cell', None):
            return '# optimized_cell:\n' + str(self.optimized_cell) + self.log
        if getattr(self, 'log', None):
            return self.log
        else:
            return 'moonphase is not 2, nothing here'   '''


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
class Electron(object):

    def __init__(self, node):

        self.gen = node.gen
        self.path = node.path
        self.prev = node.prev


    def compute(self):

        if not getattr(self, 'log', None):
            if os.path.isdir(self.path):
                raise shared.CustomError(self.__class__.__name__ + ' compute: self.path {%s} taken' %self.path)
            shutil.copytree(self.prev.path, self.path)
            os.chdir(self.path)

            if self.gen.parse_if('cell'):
                with open('POSCAR','r') as infile:
                    self.cell = Cell(infile.read())
                    
            if self.gen.parse_if('grepen'):
                self.grepen = Grepen(self.prev.gen)

            if self.gen.parse_if('dos'):
                self.dos = Dos(self.grepen)

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

    '''def __str__(self):
        if getattr(self, 'log', None):
            return self.log
        else:
            return 'moonphase is not 2, nothing here'
            
    def delete(self):
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
    '''

  
class Grepen(object):
    def __init__(self, prev_gen):
        self.log=''
        self.energy=float(os.popen('grep "energy without" OUTCAR | tail -1 | awk \'{print $5}\'').read())
        self.efermi=float(os.popen('grep "E-fermi" OUTCAR | awk \'{print $3}\'').read())

        self.nbands=int(os.popen('grep NBANDS OUTCAR | awk \'{print $15}\'').read())
        self.nedos=int(os.popen('grep NEDOS OUTCAR | awk \'{print $6}\'').read())

        self.spin = prev_gen.getkw('spin')
        self.ismear = int(prev_gen.getkw('ismear'))
        self.sigma = 0 if self.ismear!=0 else float(prev_gen.getkw('ismear'))

    
class Dos(object):
    def __init__(self,grepen):  
        
        self.log = '\n\n\n'
        self.log += '*' * 30 + ' ' + self.__class__.__name__ + ' @ ' + os.getcwd() + ' ' + '*' * 30 + '\n'
        with open('DOSCAR','r') as doscar_file:
            l = doscar_file.readlines()
            if not len(l) >= 7:
                raise shared.CustomError( 'dos.py warning: doscar is not usable (as determined by grepen).')

        ## parameter: min_dos: is dos considered 0 if it's 0.002? 
        min_dos = 1E-3

        ## self.dos
        doscar_file = open("DOSCAR","r")
        doscar_lines = doscar_file.readlines()
        doscar_lines_split = [doscar_lines[i].split() for i in range(6,6+grepen.nedos)]
        self.dos = np.float64(doscar_lines_split)   # self.dos
        idx_fermi = abs(self.dos[:,0] - grepen.efermi).argmin() + 1

        if grepen.spin == 'ncl':
            print self.__class__.__name__ + ' __init__ warning: spin=ncl is not well supported'
        if grepen.spin == 'para' or grepen.spin == 'ncl': 

            if abs(self.dos[idx_fermi][1]) > min_dos:
                self.log += 'dos.py: conductor.\n'
            else: 
                self.VB=self.dos[idx_fermi:0:-1]
                self.CB=self.dos[idx_fermi:len(self.dos)]
                self.VB1=[self.VB[x][0] for x in range(0,len(self.VB)) if abs(self.VB[x][1])>min_dos]
                self.CB1=[self.CB[x][0] for x in range(0,len(self.CB)) if abs(self.CB[x][1])>min_dos]
                if len(self.VB1)==0 or len(self.CB1)==0:
                    self.log += 'dos.py: weird. len(self.VB1/self.CB1) is 0\n'
                    exit(1)
                self.VBM1 = self.VB1[0]
                self.CBM1 = self.CB1[0]
                self.log += 'dos.py: DOS* type is insulator. DOS bandgap* is: ' + str(self.CBM1-self.VBM1) + ' eV.\n'
        elif grepen.spin=='fm' or grepen.spin=='afm':
            if abs(self.dos[idx_fermi][1])>min_dos and abs(self.dos[idx_fermi][2])>min_dos:
                self.log += 'dos.py: conductor. quite probably. but check dos anyway.\n'
            elif abs(self.dos[idx_fermi][1])<min_dos and abs(self.dos[idx_fermi][2])<min_dos: 
                self.VB=self.dos[idx_fermi:0:-1]
                self.CB=self.dos[idx_fermi:len(self.dos)]
                self.VB1=[self.VB[x][0] for x in range(0,len(self.VB)) if abs(self.VB[x][1])>min_dos]
                self.VB2=[self.VB[x][0] for x in range(0,len(self.VB)) if abs(self.VB[x][2])>min_dos]
                self.CB1=[self.CB[x][0] for x in range(0,len(self.CB)) if abs(self.CB[x][1])>min_dos]
                self.CB2=[self.CB[x][0] for x in range(0,len(self.CB)) if abs(self.CB[x][2])>min_dos]
                if len(self.VB1)==0 or len(self.VB2)==0 or len(self.CB1)==0 or len(self.CB2)==0:
                    raise shared.CustomError( 'dos.py: weird. len(self.VB1) is ' + str(len(self.VB1)) + '. len(self.VB2) is ' + str(len(self.VB2)) + '. len(self.CB1) is ' + str(len(self.CB1)) + '. len(self.CB2) is ' + str(len(self.CB2)))
                self.VBM1=self.VB1[0] ; self.CBM1=self.CB1[0] ; self.VBM2=self.VB2[0] ; self.CBM2=self.CB2[0]
                CV_divide=0.45
                self.VBM1S=self.VBM1*(1-CV_divide)+self.CBM1*CV_divide ; self.CBM1S = self.CBM1*(1-CV_divide) + self.VBM1*CV_divide ; self.VBM2S = self.VBM2*(1-CV_divide) + self.CBM2*CV_divide ; self.CBM2S = self.CBM2*(1-CV_divide) + self.VBM2*CV_divide
                if abs(self.VBM1-self.VBM2)<min_dos or abs(self.CBM1-self.CBM2)<min_dos:
                    self.log += 'dos.py: DOS* type is nonmagnetic insulator. Bandgap* is: ' + str(self.CBM1-self.VBM1) + ' eV.\n'
                else:
                    self.log += 'BMS ' + str(min(abs(self.VBM1-self.VBM2)) + str(min(self.VBM1,self.VBM2)-max(self.CBM1,self.CBM2)) + str(abs(self.CBM1-self.CBM2))) + '\n'
            elif abs(self.dos[idx_fermi][1])<min_dos and abs(self.dos[idx_fermi][2])>min_dos:
                self.VB=self.dos[idx_fermi:0:-1]
                self.CB=self.dos[idx_fermi:len(self.dos)]
                self.VBM1=next(self.VB[x][0] for x in range(0,len(self.VB)) if abs(self.VB[x][1])>min_dos or abs(self.VB[x][2])<min_dos)
                self.CBM1=next(self.CB[x][0] for x in range(0,len(self.CB)) if abs(self.CB[x][1])>min_dos or abs(self.CB[x][2])<min_dos)
                self.log += 'HM ' + str(self.CBM1-self.VBM1) + '\n'
            elif abs(self.dos[idx_fermi][1])>min_dos and abs(self.dos[idx_fermi][2])<min_dos:
                self.VB=dos[idx_fermi:0:-1]
                self.CB=dos[idx_fermi:len(self.dos)]
                self.VBM1=next(self.VB[x][0] for x in range(0,len(self.VB)) if abs(self.VB[x][2])>min_dos or abs(self.VB[x][1])<min_dos)
                self.CBM1=next(self.CB[x][0] for x in range(0,len(self.CB)) if abs(self.CB[x][2])>min_dos or abs(self.CB[x][1])<min_dos)
                self.log += 'HM ' + str(self.CBM1-self.VBM1) + '\n'

        # site-projected dos: split doscar and convert to self.site_dos
        split_indices = [i for i, x in enumerate(doscar_lines) if x == doscar_lines[5]]
        split_indices.append(len(doscar_lines))
        self.site_dos = []
        for i in range(0,len(split_indices)-1):
            l = doscar_lines[split_indices[i]+1:split_indices[i+1]]
            self.site_dos.append(np.float_([x.split() for x in l]))

        self.log += '*' * 30 + ' ' + self.__class__.__name__ + ' @ ' + os.getcwd() + ' ' + '*' * 30 + '\n'
        print self.log

# imports bandstructure from EIGENVAL. 
# imports kpoints list.
# interpolates.
# finds all sources of errors in bandstructure.
class Bands(object):

    def __init__(self,grepen):  

        # initialize
        self.log = '\n\n\n'
        self.log += '*' * 30 + ' ' + self.__class__.__name__ + ' @ ' + os.getcwd() + ' ' + '*' * 30 + '\n'
        eigenval_file = open("EIGENVAL","r")
        eigenval_lines = eigenval_file.readlines()
        if (len(eigenval_lines)<7):
            raise shared.CustomError(self.__class__.__name__ + ' __init__: EIGENVAL file is not usable')
        eigenval_lines = [eigenval_lines[i].split() for i in range(6,len(eigenval_lines))]
        nkpts = len(eigenval_lines)/(grepen.nbands+2)
        list_band_kpte=[]

        with open('KPOINTS','r') as kpoints:
            if (len(kpoints.readlines())>7):
                raise shared.CustomError(self.__class__.__name__ + ' __init__: KPOINTS does not form a mesh. Module would not work.')

        # Forking: EIGENVAL format depends on spin. 
        if grepen.spin != 'para':
            self.log += self.__class__.__name__ + " __init__ warning: only the first spin direction. We are essentially assuming spin=para.\n"
        if grepen.spin == 'para' or grepen.spin == 'ncl':
            eigenval_e_idx = 1
            eigenval_occ_idx = 2
        else:
            eigenval_e_idx = 1
            eigenval_occ_idx = 3

        # initialise all bands. self.bands[i_band][i_kpt] = [kx,ky,kz,e,occupancy]
        for i_band in range(0,grepen.nbands):
            band_kpte = []
            for i_kpt in range(0,nkpts):
                kpte = eigenval_lines[i_kpt*(grepen.nbands+2)+1][0:3]
                energy = float(eigenval_lines[i_kpt*(grepen.nbands+2)+i_band+2][eigenval_e_idx])
                occ = 1 if energy < grepen.efermi else 0
                kpte.append(energy)
                kpte.append(occ)
                band_kpte.append(kpte)
            list_band_kpte.append(band_kpte)
        self.bands = np.float64(list_band_kpte)
        ## initialise kpoints
        kpts = [kpte[0:3] for kpte in self.bands[0]]
        ### get kpts nearest neighbor list
        min_kpt_dist = np.amin(spatial.distance.pdist(kpts))
        kpts_nn_tree = spatial.cKDTree(kpts)
        kpts_nn_list = kpts_nn_tree.query_pairs(r=min_kpt_dist*1.5,output_type='ndarray')
        self.log += "bands.py: finite kpoint mesh precision delta_k (2pi/a) is %.5f; number of kpoints sampled is %d\n" %(min_kpt_dist, len(kpts))
        ## compute bandstructure-wise CBM and VBM
        self.flat_bands=[] # flattened out:
        for band in self.bands:
            for kpte in band:
                self.flat_bands.append(kpte)
        self.flat_bands=np.float64(self.flat_bands)
        CBM1_kpte_idx=np.argmin([kpte[3] for kpte in self.flat_bands if kpte[4]==0])
        VBM1_kpte_idx=np.argmax([kpte[3] for kpte in self.flat_bands if kpte[4]==1])
        self.CBM1_kpte=[kpte for kpte in self.flat_bands if kpte[4]==0][CBM1_kpte_idx]
        self.VBM1_kpte=[kpte for kpte in self.flat_bands if kpte[4]==1][VBM1_kpte_idx]
        self.CBM1=self.CBM1_kpte[3]
        self.VBM1=self.VBM1_kpte[3]
        CV_divide=0.5
        self.VBM1S=self.VBM1*(1-CV_divide)+self.CBM1*CV_divide ; self.CBM1S = self.CBM1*(1-CV_divide) + self.VBM1*CV_divide 
        self.log += "bands.py: bandstructure bandgap* is %.5f, CBM1* is %s, VBM1* is %s\n" %(self.CBM1-self.VBM1,self.CBM1_kpte[0:4],self.VBM1_kpte[0:4])
        ## compute neargap bands
        self.neargap_bands=[]
        for band in self.bands:
            if any([(kpte[3]<self.VBM1S and kpte[3]>self.VBM1-0.5) for kpte in band]) or any([(kpte[3]>self.CBM1S and kpte[3]<self.CBM1+0.5) for kpte in band]):   
                self.neargap_bands.append(band)
        self.neargap_bands=np.float_(self.neargap_bands)
        self.log += 'bands.py: number of neargap_bands is %d\n' %(len(self.neargap_bands))

        # precision check 
        ## calculate DeltaE_KPOINTS by grabbing average E diff / average E diff near bandgap from EIGENVAL.
        ### specify ranges to look for
        range_avg_kpt_de = [0.1,0.15,0.2,0.5]
        ### pretty print
        widgets = ['precision check bands: ', Percentage(), ' ', Bar(), ' ', ETA()] #pretty print
        pbar = ProgressBar(widgets=widgets, maxval=len(self.neargap_bands)).start()
        ### loop over each band (otherwise mixing is undesirable)
        avg_kpt_de=np.float_([0]*len(range_avg_kpt_de))
        count_avg_kpt_de=np.float_([0]*len(range_avg_kpt_de))
        for i_band,band in enumerate(self.neargap_bands):
            #### for each NN pair, compute |delta_e| if energy is within bound
            for nn_pair in kpts_nn_list:
                ee_pair = band[nn_pair][:,3]
                for idx,val in enumerate(range_avg_kpt_de):
                    if all([self.VBM1-val<i<self.VBM1S for i in ee_pair]) or all([self.CBM1S<i<self.CBM1+val for i in ee_pair]): 
                        avg_kpt_de[idx] += abs(ee_pair[0]-ee_pair[1])
                        count_avg_kpt_de[idx] += 1
            pbar.update(i_band+1) #pretty print
        pbar.finish() #pretty print
        for idx,val in enumerate(count_avg_kpt_de):
            if val==0:
                count_avg_kpt_de[idx]=1E-8  
        avg_kpt_de=np.divide(avg_kpt_de,count_avg_kpt_de)
        self.log += 'bands.py: NN kpoints energy delta_E = E_j\'-E_j is:\n'
        for idx,val in enumerate(range_avg_kpt_de):
            self.log += '  CBM/VBM +- %.2f eV, difference is %.5f; number of samples is %d.\n' %(val,avg_kpt_de[idx],count_avg_kpt_de[idx])
        self.de_kpoints = max(avg_kpt_de)
        ## fit the band for i) verifying smoothness ii) estimating bandgap
        widgets = ['fitting each band: ', Percentage(), ' ', Bar(), ' ', ETA()] #pretty print
        pbar = ProgressBar(widgets=widgets, maxval=len(self.neargap_bands)).start()
        self.fit_neargap_bands = []
        for i_band,band in enumerate(self.neargap_bands):
            pbar.update(i_band)
            fit_neargap_band = Rbf(band[:,0],band[:,1],band[:,2],band[:,3])
            self.fit_neargap_bands.append(fit_neargap_band)
        pbar.finish()
        ### ii) estimate bandgap
        #### in each kpoint, get a bandgap (for each fit, get a max/min, then get the band). get the global bandgap. 
        widgets = ['interpolating bandgap: ', Percentage(), ' ', Bar(), ' ', ETA()] #pretty print
        pbar = ProgressBar(widgets=widgets, maxval=len(kpts)).start()
        fit_1kpt_bandgaps=[]
        for i_kpt,kpt in enumerate(kpts):
            if i_kpt % 100 == 0:
                pbar.update(i_kpt)
            near_kpt_maxmin_bnd=[[x-min_kpt_dist/2,x+min_kpt_dist/2] for x in kpt]
            near_kpt_maxmin_energies = []
            for (i_fit_neargap_band,fit_neargap_band) in enumerate(self.fit_neargap_bands):
                if fit_neargap_band(kpt[0],kpt[1],kpt[2]) < self.VBM1S:
                    fun = lambda x: -1*fit_neargap_band(x[0],x[1],x[2]) 
                    near_kpt_maxmin_energy = -1 * minimize(fun,kpt,bounds=near_kpt_maxmin_bnd).fun
                    near_kpt_maxmin_energies.append(near_kpt_maxmin_energy)
                elif fit_neargap_band(kpt[0],kpt[1],kpt[2]) > self.CBM1S:
                    fun = lambda x: fit_neargap_band(x[0],x[1],x[2]) 
                    near_kpt_maxmin_energy = minimize(fun,kpt,bounds=near_kpt_maxmin_bnd).fun
                    near_kpt_maxmin_energies.append(near_kpt_maxmin_energy)
                else: 
                    print 'bands.py initialisation error: fit_neargap_band ',kpt,' ',fit_neargap_band(kpt[0],kpt[1],kpt[2]),' energy is not above bands.VBM1s or below bands.CBM1S. ignoring.'
                if self.VBM1 < near_kpt_maxmin_energy < self.CBM1:
                    i_energy = min(self.neargap_bands[:,i_kpt,3], key=lambda x:abs(x-near_kpt_maxmin_energy))
                    o_energy = near_kpt_maxmin_energy
                    # print 'band.py: interpolated eigenstate found. Energy* is %.4f -> %.4f eV; kpoint* is %s -> %s' %(i_energy,o_energy,kpt,minimize(fun,kpt,bounds=near_kpt_maxmin_bnd).x)
            near_kpt_maxmin_energies.append(self.VBM1)
            near_kpt_maxmin_energies.append(self.CBM1)
            fit_1kpt_bandgap = min([e for e in near_kpt_maxmin_energies if e > self.CBM1S]) - max([e for e in near_kpt_maxmin_energies if e < self.VBM1S])
            fit_1kpt_bandgaps.append(fit_1kpt_bandgap)
        self.fit_bandgap = min(fit_1kpt_bandgaps)
        pbar.finish()
        self.log += "bands.py: fitted bandgap* is %.5f. Usually bandgap is between fitted and raw bandgap*. For errors see errors.py.\n" %(self.fit_bandgap)
        self.log += '*' * 30 + ' ' + self.__class__.__name__ + ' @ ' + os.getcwd() + ' ' + '*' * 30 + '\n'

        print self.log
    
    # plot E(KPOINT)
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
    def __init__(self, cell, grepen, dos):  
        self.log = '\n\n\n'
        self.log += '*' * 30 + ' ' + self.__class__.__name__ + ' @ ' + os.getcwd() + ' ' + '*' * 30 + '\n'
        # sanity check
        if len(dos.site_dos) < 2 or len(dos.site_dos[1]) < 10:
            raise shared.CustomError(self.__class__.__name__ +' __init__: site-project DOSCAR too short.')
        if len(dos.site_dos[1][0]) not in [10,19,37]:
            raise shared.CustomError(self.__class__.__name__ +' __init__: we only support up-to-and-only-d compound. fixing is easy. exiting.')

        # let's start!
        # pristine electronic configuration
        self.log += '\n\nGS electron configurations for elements in POSCAR\n'   # a good idea is to indent within header
        for element in cell.stoichiometry:
            self.log += element + ': ' + shared.ELEMENTS[element].eleconfig + '\n'

        # integrating site-projected pdos
        self.log += '\n\nIntegrated Projected DOS: integration of DOS of wavefunctions projected onto spherical harmonics within spheres of a radius RWIGS\n'
        if grepen.spin == 'ncl':
            self.nspin = 4 # used in: i * nspin + j
            spins = 'tot x y z'.split()
        elif grepen.spin == 'fm' or grepen.spin == 'afm':
            self.nspin = 2
            spins = 'UP DN'.split()
        else:
            self.nspin = 1
            spins = ['tot']
        orbitals = 's p_y p_z p_x d_xy d_yz d_z2 d_xz d_x^2-y^2'.split()
        
        for idx_element, element in enumerate(cell.stoichiometry.keys()):
            for idx_atom in range( sum(cell.stoichiometry.values()[0:idx_element]), sum(cell.stoichiometry.values()[0:idx_element+1]) ):
                for idx_spin in range(0, self.nspin):
                    self.log += element + str(idx_atom) + '_' + spins[idx_spin]+'\t'
                    pdos = dos.site_dos[idx_atom+1]
                    idx_fermi = (np.abs(pdos[:,0]-grepen.efermi)).argmin()
                    idx_top_integral = (np.abs(pdos[:,0]-grepen.efermi-5)).argmin()
                    for idx_orbital,orbital in enumerate(orbitals):
                        integral = np.trapz(pdos[:idx_fermi,self.nspin * idx_orbital + idx_spin + 1],x=pdos[:idx_fermi,0])
                        integral = abs(integral)
                        maxintegral = np.trapz( pdos[:idx_top_integral, self.nspin * idx_orbital + idx_spin + 1], x = pdos[:idx_top_integral, 0])
                        self.log += orbital + ' ' + str('{0:.2f}'.format(integral)) + ' '
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
            
            self.log += '*' * 30 + ' ' + self.__class__.__name__ + ' @ ' + os.getcwd() + ' ' + '*' * 30 + '\n'
    
        print self.log


class Errors(object):
    def __init__(self,Agrepen,Ados,Abands,Bgrepen=None,Bdos=None,Bbands=None):
        
        self.log = '\n\n\n'
        self.log += '*' * 30 + ' ' + self.__class__.__name__ + ' @ ' + os.getcwd() + ' ' + '*' * 30 + '\n'
        ## self.rules (aka error types) in dirA
        self.rules = []
        self.de = 0
        if Agrepen.ismear == 0:
            self.de_sigma = Agrepen.sigma * 2
            if self.de_sigma > self.de: 
                self.de = self.de_sigma
            rule='gaussian smearing smoothes out irregularities with size sigma: sigma[%.4f] < DE[%.4f]/2' %(Agrepen.sigma,self.de)
            self.rules.append(rule)
        ## rule
        self.de_kpoints = Abands.de_kpoints * 2
        if self.de_kpoints > self.de:
            self.de = self.de_kpoints
        rule='sparse kpoints grid may miss in-between eigenvalues. E(j)-E(j\')[%.4f] < DE[%.4f]/2' %(Abands.de_kpoints,self.de)
        self.rules.append(rule)
        ## rule
        self.de_nedos = 10.0/float(Agrepen.nedos) * 2
        if self.de_nedos > self.de:
            self.de = self.de_nedos
        rule='all details between two DOS points are lost. 10/NEDOS[%.4f] < DE[%.4f]/2' %(10.0/Agrepen.nedos,self.de)
        self.rules.append(rule)
        ## rule
        if 10.0/float(Agrepen.nedos) < Abands.de_kpoints:
            rule='error! '
        else:
            rule=''
        rule += 'DOS should not be so fine that kpoint mesh coarseness is obvious. 10/NEDOS[%.4f] > DE_KPOINTS[%.4f]' %(10.0/Agrepen.nedos,Abands.de_kpoints)
        self.rules.append(rule)

        # comparing against dirB
        if not Bgrepen:
            self.log += 'errors.py: skipping E(k) numerical error (interpolated) comparison check. this is usually okay since such errors should be smaller than 0.01 eV. '
        else: 
            self.de_interpd = 0
            for i_band,band in enumerate(Bbands.neargap_bands):
                fitband=[[kpt_e[0],kpt_e[1],kpt_e[2],float(Abands.fit_neargap_bands[i_band](kpt_e[0],kpt_e[1],kpt_e[2])),kpt_e[4]] for kpt_e in band]
                for kpt_e in band:
                    self.log += str(kpt_e[0]) + str(kpt_e[1]) + str(kpt_e[2]) + str(kpt_e[3]) + str(float(Abands.fit_neargap_bands[i_band](kpt_e[0],kpt_e[1],kpt_e[2]))) + '\n'
                fitband = np.float_(fitband)
                band = np.float_(band)
                band_deviation = abs((band - sum(band)/len(band)) - (fitband - sum(fitband)/len(fitband)))
                avg_deviation = sum(band_deviation[:,3])/len(band_deviation[:,3])
                self.de_interpd += avg_deviation
                self.log += 'check_interpolate.py: in band %d, between dirA and dirB, interpolation plus Cauchy error is %.5f.\n' %(i_band,avg_deviation)
            ## rule
            self.de_interpd /= len(Bbands.neargap_bands)
            if Agrepen.ismear == 0 and Agrepen.sigma * 3 < self.de_interpd:
                rule='error! '
            else:
                rule=''
            rule=rule+'smearing should be larger than numerical energy error: sigma[%.4f] > DE_INTERPD[%.4f]' %(Agrepen.sigma,self.de_interpd)
            self.rules.append(rule)

        # printing self.rules
        self.log += 'errors.py: you should expect an error around %.4f eV in dirA. see details below.\n' %(self.de)
        for rule in self.rules:
            self.log += ' '*4 + rule + '\n'

        self.log += '*' * 35 + ' charge of ' + os.getcwd() + ' ' + '*' * 35 

        print self.log
