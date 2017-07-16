#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Part of qchem
# Implement require grammar
# 
# 输入有效要求:
# maintain kw_legal_set, kw_internal_set, mod_legal_set
# parse_require: if in argv flow, add not. else, add.
# parse_if: if in argv flow, add not. if in if flow, add kw not. else, add.
# 实现: 血月系统: self.moonphase=0(argv)|1(if)|2(require)|3(reshuffle)

import os
import sys
import re
import numpy as np
import scipy


class Gen(object):   # Stores the logical structure of keywords and modules. A unique construct deserving a name.

    def evaluate(self,expression):                  # Parses a literal value or (funcname)
        if expression.startswith('(') and expression.endswith(')'): 
            func = getattr(self, re.sub('\(|\)','',expression).strip())
            if not func:
                print self.__class__.__name__+' error: Bad Grammar. Unable to find function {%s}' % re.sub('\(|\)','',expression)
            return func()
        else:   # literal
            return expression

    def parse_require(self,expression,run=False):   # Parses single expression. Accepts empty expression.
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

    def parse_if(self,expression):  # recursively parse complex if condition. accepts empty expression.
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

    def __init__(self,argv): # argv is a string like 'generic opt hse06 prec=Low'
	# 读mod, kw
        self.mod = {}
	self.kw = {}
        self.kw_legal_set = set()
        self.kw_internal_set = set()
        self.mod_legal_set = set()
        self.moonphase = 0
        argv = argv.split()
        for item in argv:
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
        for name in self.mod.keys():
            if self.mod[name] == set([True]):
                self.mod[name] = True
        for name in self.kw:
            if len(self.kw[name]) != 1:
                print self.__class__.__name__+' error: non-unique output. Kw[%s]={%s} has not been restricted to 1 value.' %(name,self.kw[name]) ; sys.exit(1)
            else:
                self.kw[name] = next(iter(self.kw[name]))
        self.moonphase=3

    # misc methods for use with require
    def getkw(self,kwname):
	if len(self.kw[kwname])!=1:
	    raise ValueError('self.kw[kwname] does not have 1 and only 1 value. wait till that happens and try again.')
	return next(iter(self.kw[kwname]))
    ## methods that return value
    ## methods that return boolean
    def ismear5check(self):
	try:
	    kpoints = self.getkw('kpoints').split(' ')
	    return np.prod(map(int,kpoints)) > 2
	except (KeyError, AttributeError, ValueError) as KwError:
	    return False
    def nkred_divide(self):
	try:
	    kpoints = map(int,self.getkw('kpoints').split(' '))
	    nkredx = int(self.getkw('nkredx'))
	    nkredy = int(self.getkw('nkredy'))
	    nkredz = int(self.getkw('nkredz'))
	    return kpoints[0]%nkredx==0 and kpoints[1]%nkredy==0 and kpoints[2]%nkredz==0
	except (KeyError, AttributeError, ValueError) as KwError:
	    return False
    ## methods that writes files
    def compute(self):
	with open('INCAR','w') as of:
	    for name in self.kw:
                if name not in self.kw_internal_set:
	            of.write('\t'+name.upper()+' \t= '+self.kw[name]+'\n')
        # to be continued!


# =========================================================================== 

# Pos: stores and parses poscar.

class Poscar(object):
    def __init__(self):
        f=open("POSCAR","r")
        self.lines=f.readlines()
        self.cell=[self.lines[i].split() for i in range(2,5)]
        self.base=[self.lines[i].split()[0:3] for i in range(8,len(self.lines))]
        self.elements = self.lines[5].split()
        self.nelements = len(self.elements)
        self.atomcounts = np.int_(self.lines[6].split())
        self.natoms = sum(self.atomcounts)
        self.cell=np.float64(self.cell)
        self.base=np.float64(self.base)
        f.close()
        with open('POSCAR','r') as infile:
            self.lines = infile.read().splitlines()
            self.name = self.lines[0]
            self.cell = np.float_([ line.split() for line in lines[2:5] ]) * float(lines[1])
            if 
            self.elements = dict( zip(lines[5].split(), map(int,lines[6])) )
            if not lines[7].startswith('D'):
                print 'pre.poscar error: unsupported format. Only direct coordinates are supported. Explanation: who the heck uses Cartesian?'

