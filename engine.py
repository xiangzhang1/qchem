#!/usr/bin/python
# -*- coding: utf-8 -*-

# Regarding import: 
# `import phase` is forbidden. Check for cyclic import.
#
# Regarding errors and warnings:
# Be flexible between print and raise. Use self.__class__.__name__
#
# Regarding __init__, compute, and write_:
# Do not define compute() unless necessary.
#
# ================================================================================
#
# Gen

import os
import sys
import subprocess
import re
import numpy as np
import scipy
import shutils

from phase import ELEMENTS


class Gen(object):   # Stores the logical structure of keywords and modules. A unique construct deserving a name.

    # Utilities
    # ---------
    def getkw(self,kwname):
	if len(self.kw[kwname])!=1:
	    raise ValueError('self.kw[kwname] does not have 1 and only 1 value. wait till that happens and try again.')
	return next(iter(self.kw[kwname]))

    def evaluate(self,expression):                  # Evaluates expression to string: literal or (funcname)
        if expression.startswith('(') and expression.endswith(')'): 
            func = getattr(self, re.sub('\(|\)','',expression).strip())
            if not func:
                print self.__class__.__name__+' error: Bad Grammar. Unable to find function {%s}' % re.sub('\(|\)','',expression)
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
            if run and not bool(result):
                print 'Target %s={%s}. ' %(kwname,self.kw[kwname] if kwname in self.kw else 'null')
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
            return self.parse_if(expression)

    def parse_if(self,expression):  # recursively evaluate complex if condition. accepts empty expression.
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

    # main
    # -------------
    def __init__(self,input_,cell): 
        self.cell = cell
	# 读mod, kw
        self.mod = {}
	self.kw = {}
        self.kw_legal_set = set()
        self.kw_internal_set = set()
        self.mod_legal_set = set()
        self.moonphase = 0
        input_ = input_.split()
        for item in input_:
            self.parse_require(item,True)
	# 执行require
        self.require = []
        self.moonphase = 1
	with open('engine.gen.conf') as conf:
	    lines = conf.read().splitlines()
            for line in [ [p.strip() for p in l.split(':')] for l in lines if not l.startswith('#') ]:
                if len(line) < 3:
                    print self.__class__.__name__+' error: Bar Grammar. Require line format wrong: {%s}' % line ; sys.exit(1)
                for part in [p.strip() for p in line[1].split(',') ]:
                    if self.parse_if(line[0]) and self.parse_require(part,False) and line[2]=='optional':
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
                    sys.stdout.write( self.__class__.__name__+' warning: conflict. Requirement is { %s : %s }. Explanation: { %s }.' % (line[0],line[1],line[3]) )
                else:
                    sys.stdout.write( self.__class__.__name__+' error: conflict. Requirement is { %s : %s }. Explanation: { %s }.' % (line[0],line[1],line[3]) )
                if '=' not in line[1]:  ## cariage return. 
                    sys.stdout.write( "\n" );
                self.moonphase=2 ; self.parse_require(line[1],True) ; self.moonphase=1
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
                print self.__class__.__name__+' error: non-unique output. Kw[%s]={%s} has not been restricted to 1 value.' %(name,self.kw[name]) ; sys.exit(1)
        self.moonphase=3    # parsing and validating of input_ is complete.
        self.check_memory()

    def write_(self):
        if self.moonphase != 3:
            raise SyntaxError('write_ must be called in blood moon.')
        if self.parse_if('vasp'):
    	    with open('INCAR','w') as of_:
    	        for name in self.kw:
                    if name not in self.kw_internal_set:
    	                of_.write('\t'+name.upper()+' \t= '+self.getkw(name)+'\n')
            self.cell.write_()
            for symbol in self.cell.stoichiometry.keys():
                self.write_potcar(symbol)
            self.write_kpoints(self.getkw('kpoints'))

    def write_kpoints(kpoints):
        with open('KPOINTS','w') as of_:
            of_.write('KPOINTS\n')
            of_.write('0\n')
            of_.write(kpoints.split()[3])
            of_.write( ' '.join(kpoints.split()[0:3]) + '\n' )
            of_.write('0 0 0')

    def write_potcar(symbol):
        if len(ELEMENTS[symbol].pot) == 0:
            raise ReferenceError('POTCAR for '+symbol+' not found.')
        path = os.path.dirname(os.path.realpath(__file__))
        path += '/data/paw_pbe/'+ELEMENTS[symbol].pot[0] + '/POTCAR' + ELEMENTS[symbol].pot_extension
        if_ = open(path,'r')
        of_ = open('./POTCAR','w')
        of_.write( if_.read() )

    def check_memory(self):
        if self.moonphase != 3:
            raise SyntaxError('write_ must be called in blood moon.')
        # make temporary dir
        os.mkdir('memory_check')
        os.chdir('memory_check')
        # alter and write
        wayback_dict = []
        if self.parse_if('isym=-1'):
            wayback_dict.add( lambda x: x.kw['isym']=['-1'] )
            multiplier = 2
            self.kw['isym'] = ['-1']
        if self.parse_if('lsorbit=.TRUE.'):
            wayback_dict.add( lambda x: x.kw['lsorbit']=['.TRUE.']
            multiplier = 2
            self.kw['lsorbit'] = ['.FALSE.']
        self.write_()
        for funcname in wayback_dict:
            funcname(self)
        os.system("sed -i '6d' POSCAR")
        # calculate and read
        path = os.path.dirname(os.path.realpath(__file__))
        output = subprocess.check_output([path+'/data/makeparam']).splitlines()
        try:
            self.memory = {}
            self.memory['arraygrid'] = int( next(l for l in output if 'arrays on large grid' in l).split()[7] )
            self.memory['wavefunction']  = int( next(l for l in output if 'sets of wavefunctions' in l).split()[4] )
            self.memory['projector_real']  = abs(int( next(l for l in output if 'projectors in real space' in l).split()[4] ))
            self.memory['projector_reciprocal']  = abs(int( next(l for l in output if 'projectors in reciprocal space' in l).split()[4] ))
        except StopIteration, KeyError:
            print self.__class__.__name__ + 'error: makeparam output illegal. Check POSCAR4 format and memory leak.'
            raise SystemError
        # parse and raise error
        if self.parse_if('hse'):
            memory_required = ( (memory['projector_real']+memory['project_reciprocal'])*self.getkw('npar')+3*memory['wavefunction']*self.getkw('kpar') )/1024^3 + self.getkw('nodes')
        else:
            memory_required = ( (memory['projector_real']+memory['project_reciprocal'])*self.getkw('npar')+memory['wavefunction']*self.getkw('kpar') )/1024^3 + self.getkw('nodes')*3/2
        memory_required *= multiplier
        memory_availble = int(self.getkw('nodes')) * int(self.getkw('mempernode')
        if memory_required > memory_available:
            print self.__class__name + 'warning: insufficient memory. Mem required is {' + str(memory_required) + '} GB. Available mem is {' + memory_available + '} GB.' 
        # cleanup
        os.chdir('..')
        shutil.rmtree('memory_check')

    # User-defined (funcname)
    # -----------------------
    def totalnumbercores(self):
        return str( int(self.getkw('nodes')) * int(self.getkw('corespernode')) )

    def nbands(self):
        print self.__class__.__name__ + ' warning: nbands may not be that reliable'
        if self.parse_if('spin=ncl'):
            nbands = ( self.cell.nelect * 3 / 5 + self.cell.nion * 3 / 2 ) * 2
        elif self.parse_if('spin=para'):
            nbands = self.cell.nelect * 3 / 5 + self.cell.nion * 1 / 2
        elif self.parse_if('spin=afm|spin=fm'):
            nbands = self.cell.nelect / 2 + self.cell.nion / 2
        else:
            raise SyntaxError('spin variable is not fm, afm or para, cannot compute nbands')
        # hse case when hse mod is not even defined. for ref, i think. hiya, later self.
        if self.parse_if('hse|prehf'):
            npar = int(self.getkw('npar'))
            nbands = (nbands + npar -1 ) / npar * npar 
        return nbands

    
    def lmaxmix(self):
        b_l_map = { 's': 2, 'p': 2, 'd': 4, 'f': 6, 'g': 8 }
        lmaxmix = max( [ b_l_map[ ELEMENTS[symbol].block ] for symbol in  self.cell.stoichiometry.keys() ] )
        return lmaxmix

    def ismear5check(self):
	try:
	    kpoints = self.getkw('kpoints').split(' ')
	    return np.prod(map(int,kpoints)) > 2
	except (KeyError, AttributeError, ValueError) as KwError:
	    return False

    def kpointscheck(self):
        if len(kpoints.split()) != 4 or not kpoints.split()[3].startswith(('G','g','M','m')):
            print self.__class__.__name__ + ' error: bad kpoints format. kpoints should be "kx ky kz monkhorst|G'
            return False
        else:
            return True

    def nkred_divide(self):
	try:
	    kpoints = map(int,self.getkw('kpoints').split(' '))
	    nkredx = int(self.getkw('nkredx'))
	    nkredy = int(self.getkw('nkredy'))
	    nkredz = int(self.getkw('nkredz'))
	    return kpoints[0]%nkredx==0 and kpoints[1]%nkredy==0 and kpoints[2]%nkredz==0
	except (KeyError, AttributeError, ValueError) as KwError:
            print self.__class__.__name__ + ' error: kx should be divisible by nkredx, etc.'
	    return False

    def magmom(self):
        magmom = ''
        if self.parse_if('spin=afm'):
            print self.__class__.__name__ + ' warning: more than 1 AFM pattern exists.'
            for symbol in self.cell.stoichiometry:
                l = [0] * self.cell.stoichiometry[symbol]
                base = ELEMENTS[symbol].magmom
                l[::2] = base
                l[1::2] = -1 * base
                magmom += ' '.join(l)
        if self.parse_if('spin=fm'):
            for symbol in self.cell.stoichiometry:
                base = ELEMENTS[symbol].magmom
                magmom += str( self.cell.stoichiometry[symbol] ) + '*' + str( base )
        return magmom
    def ldauu(self):
        ldauu = ''
        for symbol in self.cell.stoichiometry:
            ldauu += str( ELEMENTS[symbol].ldauu )
        return ldauu
    def ldauj(self):
        ldauj = ''
        for symbol in self.cell.stoichiometry:
            ldauj += str( ELEMENTS[symbol].ldauj )
        return ldauj

# =========================================================================== 


# Cell: stores and parses atomic configuration file.

class Cell(object):

    def __init__(self,lines): 
        # basics
        self.name = lines[0]
        self.base = np.float_([ line.split() for line in lines[2:5] ]) * float(lines[1])
        self.coordinates = np.float_([ line.split() for line in lines[8:] ])
        self.stoichiometry = dict( zip(lines[5].split(), map(int,lines[6])) )
        if not lines[7].startswith('D'):
            raise SyntaxError('unsupported POSCAR5 format. Only direct coordinates are supported.')
        if len(lines) - 8 != sum(self.stoichiometry.values()):
            raise SyntaxError('POSCAR5 verification failed. Line count does not match stoichiometry. Blank lines?')
        # some computation
        self.nion = sum(self.stoichiometry.values())
        self.nelect = sum( [self.stoichiometry[symbol] * ELEMENTS[symbol].pot_zval for symbol in self.stoichiometry] )

    def write_(self):
        with open('POSCAR','w') as of:
            of.write(self.name+'\n')   
            of.write('1\n')
            for line in base:
                of.write(' '.join(line)+'\n')
            of.write(' '.join(self.stoichiometry.keys()))
            of.write(' '.join(self.stoichiometry.values()))
            of.write('Direct\n')
            for line in coordinates:
                of.write(' '.join(line)+'\n')
            

# =========================================================================== 

from phase import KETS  # possible cyclic import

class Property_wanted(object):  # bad name, but good for logic
    def __init__(self,input_,path_prefix):
        # nodes part and graph part
        nodes_input = input_.split('\n\n')[0].splitlines()
        graph_input = input_.split('\n\n')[1].splitlines()
        self.nodes = {}
        self.edges = {}
        # parse nodes part
        try:
            for line_ in nodes_input:
                line = [ x.strip() for x in line_.split(':') ]
                uid = line[0]
                if uid in KETS:
                    raise SyntaxError('uid already in KETS')
                text = line[1]
                path = line[2] if len(line[2])>3 else path_prefix+'/'+uid
                self.nodes[uid] = [text,path]                
                self.edges[uid] = []
            for line_ in graph_input:
                line = [ x.strip() for x in line_.split('->') ]
                from_ = line[0]
                to_ = line[1] if line[1] else None
                self.edges[from_].add(to_)
        except (KeyError, IndexError) as SyntaxError:
            print 'bad graph syntax'
        if set(self.edges.keys()) != set(self.nodes.keys()):
            raise SyntaxError('node list in self.nodes does not match self.edges')

    def edit(self,input_,path_prefix):
        self.__init__(input_,path_prefix)

    def compute(self,proposed_uid=None):
        self.update()
        list0 = [] ; list1 = [] #1 status have priority
        for x in self.nodes:
            if x in KETS and KETS[x].moonphase()==1:
                list1.add(x)
            if x not in KETS:
                parents = [y for y in self.nodes if x in self.edges[y]]
                if not parents or any([y in KETS and KETS[y].moonphase()==3 for y in parents]):
                    list0.add(x)
        result = list0 + list1
        if not result:
            print self.__class__.__name__ + ': Nothing to compute'.
        if proposed_uid and proposed_uid in result:
            return proposed_uid
        else:
            return result[0]

    def moonphase(self):
        if any( [x not in KETS or KETS[x].moonphase()==1 for x in self.nodes ] ):
            return 1
        elif any( [ KETS[x].moonphase()==2 for x in self.nodes ] ):
            return 2
        else:
            return 3



#===========================================================================  



