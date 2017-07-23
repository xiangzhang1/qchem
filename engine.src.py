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

from shared import ELEMENTS, NODES

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
            return func(self)
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
                kwvalset.append(self.evaluate(kwval_part))
            if kwname in self.kw and bool(self.kw[kwname]):
                result = operation_map[operator](self.kw[kwname],kwvalset)
            else:
                result = operation_map[operator](kwvalset,kwvalset)
            if run and bool(result):        
                self.kw[kwname] = result
            if run and not bool(result):
                print 'Target %s={%s}. ' %(kwname,self.kw[kwname] if kwname in self.kw else 'null')
            if self.moonphase>0:    self.kw_legal_set.append(kwname)
            return bool(result)
        elif 'internal' in expression:      ## parse kwname internal
            kwname = re.sub('internal','',expression).strip()
            if self.moonphase>0:  self.kw_internal_set.append(kwname)
            return True
        elif not '(' in expression and not 'null' in expression:    ## parse !modname
            modname = re.sub('!','',expression).strip()
            if '!' in expression:
                result = ( self.mod[modname] if modname in self.mod else set() ) - set([True])
            else:
                result = ( self.mod[modname] if modname in self.mod else set() ) | set([True])
            if run and bool(result):        ### output
                self.mod[modname] = result
            if self.moonphase>0:  self.mod_legal_set.append(modname)
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
            if self.moonphase==2 or (self.moonphase==1 and result):  self.kw_legal_set.append(name)
            return result
        elif expression.startswith('(') and expression.endswith(')'):    ## parse (funcname)
            return self.evaluate(expression)
        elif expression == '':                  ## parse empty expression
            return True
        elif 'null' in expression:              ## parse kwname null
            kwname = re.sub('null','',expression).strip()
            result = not ( kwname in self.kw and bool(self.kw[kwname]) )
            if self.moonphase==2 or (self.moonphase==1 and result):  self.kw_legal_set.append(kwname)
            return result
        else:                                   ## parse modname
            if self.moonphase>0:    self.mod_legal_set.append(expression)
            return (expression in self.mod and self.mod[expression]==set([True]))        

    def write_incar_kpoints(self):
        with open('INCAR','w') as outfile:
            for name in self.kw:
                if name not in self.kw_internal_set:
    	            outfile.write('\t'+name.upper()+' \t= '+self.getkw(name)+'\n')
        with open('KPOINTS','w') as outfile:
            outfile.write('KPOINTS\n')
            outfile.write('0\n')
            outfile.write(self.getkw(kpoints).split()[3])
            outfile.write( ' '.join(self.getkw(kpoints).split()[0:3]) + '\n' )
            outfile.write('0 0 0')

    def __str__(self):
        result = ''
        for name in self.mod:
            if self.parse_if(name):
                result += name
        for name in self.kw:
            result += name + '=' + self.getkw(name) + ' '
        return result
    

    # main
    # -------------
    def __init__(self,input_,cell): 
        self.cell = cell
        if '\n' in input_:
            input_.input_.splitlines()[0]
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

    def check_memory(self):
        # make temporary dir
        os.mkdir('memory_check')
        os.chdir('memory_check')
        # alter and write
        wayback = []
        if self.parse_if('isym=-1'):
            wayback.append( 'isym=-1' )
            multiplier = 2
            self.kw['isym'] = ['0']
        if self.parse_if('lsorbit=.TRUE.'):
            wayback_dict.append( 'lsorbit=.TRUE.' )
            multiplier = 2
            self.kw['lsorbit'] = ['.FALSE.']
        self.write_incar_kpoints()
        if 'isym=-1' in wayback:
            self.kw['isym'] = ['-1']
        if 'lsorbit=.TRUE.' in wayback:
            self.kw['lsorbit'] = ['.TRUE.']
        for funcname in wayback_dict:
            funcname(self)
        os.system("sed -i '6d' POSCAR")
        # calculate and read
        path = os.path.dirname(os.path.realpath(__file__))
        output = subprocess.check_output([path+'/resource/makeparam']).splitlines()
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
        memory_availble = int(self.getkw('nodes')) * int(self.getkw('mempernode'))
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
	    return np.prod([int(x) for x in kpoints]) > 2
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
	    kpoints = [int(x) for x in self.getkw('kpoints').split(' ')]
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
        if '\n' in lines:
            lines = lines.splitlines()
        self.name = lines[0]
        self.base = np.float_([ line.split() for line in lines[2:5] ]) * float(lines[1])
        self.coordinates = np.float_([ line.split() for line in lines[8:] ])
        self.stoichiometry = dict( zip(lines[5].split(), [int(x) for x in lines[6]]) )
        if not lines[7].startswith('D'):
            raise SyntaxError('unsupported POSCAR5 format. Only direct coordinates are supported.')
        if len(lines) - 8 != sum(self.stoichiometry.values()):
            raise SyntaxError('POSCAR5 verification failed. Line count does not match stoichiometry. Blank lines?')
        # some computation
        self.nion = sum(self.stoichiometry.values())
        self.nelect = sum( [self.stoichiometry[symbol] * ELEMENTS[symbol].pot_zval for symbol in self.stoichiometry] )

    '''def __str__(self):
        result = self.name+'\n'
        result += '1\n'
        for line in base:
            result += ' '.join(line)+'\n'
        result += ' '.join(self.stoichiometry.keys())
        result += ' '.join(self.stoichiometry.values())
        result += 'Direct\n'
        for line in coordinates:
            result += ' '.join(line)+'\n'
        return result'''
            

# =========================================================================== 

class Vasp(object):

    def __init__(self, gen, cell, path, prev):
        self.gen = gen
        self.cell = cell
        self.path = path
        self.prev = prev
        os.mkdir(self.path)


    def compute(self):

        if not getattr(self, 'wrapper'):
            if not os.path.isdir(self.path):
                os.mkdir(self.path)
            shutil.copytree(self.prev.path, self.path)
            if self.gen.parse_if('icharg=1|icharg=11'):
                shutil.copyfile(self.prev.path+'/CHGCAR', self.path+'/CHGCAR')
            if self.gen.parse_if('icharg=0|icharg=10|istart=1|istart=2'):
                shutil.copyfile(self.prev.path+'/WAVECAR', self.path+'/WAVECAR')
            # write incar etc
            os.chdir(self.path)
            self.gen.write_incar_kpoints()
            with open('POSCAR','w') as f:
                f.write(str(self.cell))
            for symbol in self.cell.stoichiometry.keys():
                self.pot(symbol)
            # setting variables
            totalnumbercores = self.gen.getkw('totalnumbercores')
            if self.gen.parse_if('spin=ncl'):   # vasp flavor
                flavor = 'ncl'
            elif all([int(x)==1 for x in self.gen.getkw('kpoints').split()[0:3]]):
                flavor = 'gam'
            elif self.gen.getkw('totalnumbercores') == '0':
                totalnumbercores = 1
                flavor = 'gpu'
                print self.__class__.__name__ + ': vasp_gpu'
            else:
                flavor = 'std'
            foldername = self.path.split('/')[-2] + '_' + self.path.split('/')[-1] + '_' + hashlib.md5(self.path)
            # write scripts and instructions
            wrapper = '#!/bin/bash\n' ; subfile = '' ; instruction = ''
            if self.gen.getkw('platform') == 'dellpc':
                subfile += 'echo $PWD `date` start; echo '+'-'*75+'\n'
                subfile += 'mpiexec.hydra -n %s /home/xzhang1/src/vasp.5.4.1/bin/vasp_%s </dev/null \n' %(totalnumbercores, flavor)
                subfile += 'mail -s "VASP job finished: {${PWD##*/}}" 8576361405@vtext.com <<<EOM'
                subfile += 'echo $PWD `date` end  ; echo '+'-'*75+'\n'
                wrapper += 'nohup ./subfile 2>&1 >> run.log &'
            if self.gen.getkw('platform') == 'nanaimo':
                wrapper += 'rsync -a . nanaimo:~/%s\n'
                wrapper += 'ssh nanaimo <<EOF\n'
                wrapper += ' cd %s\n'
                wrapper += ' sbatch --nodes=%s --ntasks=%s --job-name=%s -t 12:00:00 --export=ALL subfile' %(self.gen.getkw('nodes'),totalnumbercores,self.path)
                wrapper += 'EOF\n'
                subfile += '#!/bin/bash\n. /usr/share/Modules/init/bash\nmodule purge\nmodule load intel\nmodule load impi\nmpirun -np %s /opt/vasp.5.4.4/bin/vasp_%s' %(totalnumbercores, flavor)
            with open('wrapper','w') as of_:
                of_.write(wrapper)
                os.system('chmod +x wrapper')
            if subfile:
                with open('subfile','w') as of_:
                    of_.write(subfile)
            print self.__class__.__name__ + ': %s ready to be computed. Run wrapper or press y.'
            if raw_input() == 'y':
                os.system(wrapper)
            print self.__class__.__name__ + ': waiting for system. Leave me on, or save/load.'

        elif not getattr(self,'log',None):
            if os.path.isfile('./vasprun.xml'):
                with open('vasprun.xml','r') as if_:
                    if if_.read.splitlines()[-1] != '</modeling>' and not os.path.isfile('ignore_error'):
                        raise SystemError('Vasp computation at %s went wrong. Touch ignore_error to ignore.' %self.path)
                    else:
                        with open('run.log','r') as if_:
                            self.log = if_.read()

        else:
            print self.__class__.__name__ + ': calculation already completed at %s. Why are you here?' %self.path

    def moonphase(self):
        if not getattr(self, 'wrapper', None):
            return 0
        elif not getattr(self, 'log', None):
            self.compute()
            if not getattr(self, 'log', None):  return 1
        else:
            return 2

    def pot(symbol):
        if len(ELEMENTS[symbol].pot) == 0:
            raise ReferenceError('POTCAR for '+symbol+' not found.')
        path = os.path.dirname(os.path.realpath(__file__))
        path += '/resource/paw_pbe/'+ELEMENTS[symbol].pot[0] + '/POTCAR' + ELEMENTS[symbol].pot_extension
        if_ = open(path,'r')
        of_ = open('./POTCAR','w')
        of_.write( if_.read() )

    def delete(self):
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)


# ===========================================================================  


class Map(object):


    '''def lookup(self, name):

        if name in NODES:
            return NODES.pop(name)
        elif any([x.name == name for x in self.map]):
            return next([x for x in self.map if x.name == name])
        elif '.' in name:
            return self.lookup('.'.join(name.split('.')[:-2])).map.lookup(name.split('.')[-1])
        elif name == 'master':
            if name in NODES:   return NODES['master']
            else: raise SystemError('找不到master，求喂食')
        else:
            raise LookupError('Node %s not found' %name)


    def prev(self, node):
        l = [x for x in self._map if node in x]
        if len(l) > 1: 
            raise LookupError('%s has more than 1 parent node. (wtf?)' %name)
        elif len(l) == 1:
            return l[0]
        else:
            return None'''


    def __init__(self, text):
    
        '''self._dict, self._dict2 = {}
        if '\n' in text:
            text = text.splitlines()'''

        # src -> dst
        for line in text:
            '''line = [x.strip() for x in re.split('\(->\|-->\)', line)]'''
            if len(line) == 1:
                src = self.lookup(line[0])
                self._dict[src] = []
            elif len(line) == 3:
                src, dist = self.lookup(line[0]), self.lookup(line[2])
                '''m = self._dict if line[2]=='->' else self._dict2
                m[src] = [dist] if src not in m else m[src]+[dist]'''
            else:
                raise SyntaxError('Map: src -> dst. 3 parts needed')



    '''def __str__(self):
        result = ''
        for m in self._dict, self._dict2:
            for src in m:
                for dst in m[src]:
                    result += src.name + '->' if m==self._dict else '-->' + dst.name + '\n'
                if not m[src]:
                    result += src.name + '\n'
        return result

    def __contains__(self, item):
        return item in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __getitem__(self, key):
        return self._dict[key]

    def pop(self, node):
        for m in self._map, self._map2:
            del m[node]
            for n in m:
                m[n] = [x for x in m[n] if x != node]
    '''
