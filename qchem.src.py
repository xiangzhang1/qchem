import pickle
import re
import os
import time

import engine
import shared

# ==================================================

# Node

def Import(text):

    '''if '#' not in text:
        raise shared.CustomError('qchem.Import: bad syntax. Your text is {%s}.' %text)'''
    l = re.split('^#+\s*', text, flags=re.MULTILINE) ; l.pop(0)
    l = ['# '+x for x in l]

    while l:
        # print 'Import: parsing %s' %(l[-1].splitlines()[0] if l[-1].splitlines() else '')
        n = Node(l.pop())
        '''if n.name in shared.NODES:
            raise shared.CustomError(' Import: Node name %s is in already in shared.NODES.' %n.name)'''
        shared.NODES[n.name] = n

def Dump(): 
    # distinguish data source by prefix: shared.NODES.dump, shared.NODES.markdown, sigma.dump
    # distinguish datetime by postfix: 20170803094500
    if 'master' not in shared.NODES:
        raise shared.CustomError('Dump: NODES is empty. You really should not dump.')
    with open(shared.SCRIPT_DIR + '/data/shared.NODES.dump.'+time.strftime('%Y%m%d%H%M%S'),'wb') as dumpfile:
        pickle.dump(shared.NODES, dumpfile, protocol=pickle.HIGHEST_PROTOCOL)
    print 'Dumped' + str(shared.NODES)


def Load(datetime=None):
    if datetime:
        filename = shared.SCRIPT_DIR + '/data/shared.NODES.dump.' + datetime
    else:
        l = [x for x in os.listdir(shared.SCRIPT_DIR + '/data/') if x.startswith('shared.NODES.dump')]
        if not l:   raise shared.CustomError('Load: no file to load')
        l.sort()
        filename = shared.SCRIPT_DIR + '/data/' + l[-1]
    if os.path.isfile(filename):
        with open(filename,'rb') as dumpfile:
            shared.NODES = pickle.load(dumpfile)
        print 'Loaded' + str(shared.NODES)
    else:
        raise shared.CustomError('File {%s} not found' %filename)


class Node(object):

    def __init__(self, text='# newnode'):   
        # parses 1 node at a time. searches in NODES.
        # default to a newnode

        namevalpairs = text.split('\n\n')

        # node.name
        '''if len(namevalpairs[0].splitlines()) != 1 or not namevalpairs[0].startswith('#'):
            raise shared.CustomError(self.__class__.__name__ +': __init__: Titleline format is #name:opt. Your titleline is: {%s}' %namevalpairs[0])'''
        titleline = namevalpairs.pop(0).rstrip().lstrip('# ')
        l = [x.strip() for x in titleline.split(':')]
        self.name = l[0]
        if len(l) == 2: self.property = l[1]
        
        # node.__dict__
        
        while namevalpairs:
            '''namevalpair = namevalpairs.pop(0)
            if not namevalpair.rstrip():
                continue
            name = namevalpair.split('\n')[0].strip(': ')
            if name not in shared.READABLE_ATTR_LIST: 
                continue
            value = namevalpair.split('\n',1)[1] if len(namevalpair.split('\n',1))>1 else ''
            if getattr(engine, name.title(), None):
                value = getattr(engine, name.title())(value)'''
            setattr(self, name, value)

        '''# test gen if possible
        if getattr(self,'cell',None) and getattr(self,'phase',None) and getattr(self,'property',None):
           test_gen = engine.Gen(self)'''
        

    '''def reset(self):
        # reset moonphase =1. remove all non-readable attributes.
        # remove engine
        if getattr(self, 'gen', None):
            engine_name = self.gen.getkw('engine')
            engine_ = getattr(self,self.gen.getkw('engine'),None)
            if getattr(self,self.gen.getkw('engine'),None):
                getattr(self,self.gen.getkw('engine'),None).delete()
        # remove all
        for varname in vars(self).keys():
            if varname not in shared.INPUT_ATTR_LIST:
                delattr(self, varname)
                print self.__class__.__name__ + ' reset: attribute {%s} deleted' %varname
        filename = self.path + '/.moonphase'
        if os.path.isfile(filename):
            os.remove(filename)
    '''


    @shared.moonphase_wrap
    def moonphase(self):
        if getattr(self, 'map', None):
            min_moonphase = min([x.moonphase() for x in self.map])
            max_moonphase = max([x.moonphase() for x in self.map])
            if min_moonphase > 0 or max_moonphase<1:
                return min_moonphase
            else:   # complex scenario. 1->0 counts as 1.
                if [x for x in self.map if x.moonphase()==0 and (not self.map.prev(x) or self.map.prev(x).moonphase()==2) ]:
                    return 0
                else:
                    return 1
        elif getattr(self, 'gen', None) and getattr(self, self.gen.getkw('engine'), None):
            return getattr(self, self.gen.getkw('engine')).moonphase()
        elif getattr(self, 'property', None):
            return 0
        else:
            return -2
    
            
    '''def __str__(self):
        result = '# ' + self.name + '\n\n'
        for varname in vars(self):
            if varname != 'name':
                result += varname + ':\n' + str(vars(self)[varname]) + '\n\n'
        result += '\n\n'
        return result
    '''


    def compute(self, proposed_name=None):

        if getattr(self, 'map', None):

            l = [x for x in self.map if x.moonphase()==0 and (not self.map.prev(x) or self.map.prev(x).moonphase()==2) ]
            '''if not l:
                raise shared.CustomError( self.__class__.__name__ + ': nothing to compute in parent node {%s]' %self.name )
            if proposed_name:
                tmp_l = [x for x in l if x.name == proposed_name]
                if not tmp_l:
                    raise shared.CustomError( self.__class__.__name__ + ' compute: cannot found proposed_name {%s} in map' %proposed_name)
                n = tmp_l[0]
            else:'''
                n = l[0]

            for vname in vars(self):
                if vname in shared.INHERITABLE_ATTR_LIST and not getattr(n, vname, None):
                    setattr(n,vname,getattr(self,vname))
            n.prev = self.map.prev(n)
            n.compute()

        elif getattr(self, 'property', None):
            if not getattr(self, 'cell', None) or not getattr(self, 'phase', None):
                raise shared.CustomError(self.__class__.__name__ + ' compute: cell or phase is missing. Either make sure parent has something you can inherit, or enter them.')
            if not getattr(self, 'path', None):
                self.path = raw_input('Provide path for this node: \n %s \n >:' %str(self))    # counterpart implemented in sigmajs
            if not getattr(self, 'gen', None):
                self.gen = engine.Gen(self)
            if not getattr(self, self.gen.getkw('engine'), None):
                engine_class = getattr(engine, self.gen.getkw('engine').title())
                engine_ = engine_class(self)
                setattr(self, self.gen.getkw('engine'), engine_)
            getattr(self, self.gen.getkw('engine')).compute()

        else:
            raise shared.CustomError(self.__class___.__name__ + ': compute: Node %s is not computable.' %self.name)

    def delete(self):
        if getattr(self, 'gen', None):
            engine_name = self.gen.getkw('engine')
            engine_ = getattr(self,self.gen.getkw('engine'),None)
            if getattr(self,self.gen.getkw('engine'),None):
                getattr(self,self.gen.getkw('engine'),None).delete()
        for node in engine.Map().lookup('master').map.traverse():
            if getattr(node,'map',None):
                node.map.del_node(self)
    '''

