import pickle
import re
import os
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

'''def Dump():
    with open(shared.SCRIPT_DIR + '/data/shared.NODES.dump','wb') as dumpfile:
        pickle.dump(shared.NODES, dumpfile, protocol=pickle.HIGHEST_PROTOCOL)
    print 'Dumped' + str(shared.NODES)

def Load():
    filename = shared.SCRIPT_DIR + '/data/shared.NODES.dump'
    if os.path.isfile(filename):
        with open(filename,'rb') as dumpfile:
            shared.NODES = pickle.load(dumpfile)
        print 'Loaded' + str(shared.NODES)
    else:
        raise shared.CustomError('No shared.NODES.dump file to load')
'''

class Node(object):

    def __init__(self, text):   # parses 1 node at a time. searches in NODES

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

        '''# test gen
        if getattr(self,'cell',None) and getattr(self,'phase',None) and getattr(self,'property',None):
           test_gen = engine.Gen(self.phase + ' ' + self.property, self.cell)'''
        

    '''def reset(self):
        # reset moonphase =1. remove all non-readable attributes.
        for varname in vars(self).keys():
            if varname not in shared.READABLE_ATTR_LIST:
                delattr(self, varname)
                print self.__class__.__name__ + ' reset: attribute {%s} deleted' %varname
    '''


    @shared.moonphase_wrap
    def moonphase(self):
        if getattr(self, 'map', None):
            return min([x.moonphase() for x in self.map])
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

            l = [x for x in self.map if x.moonphase()==0] + [x for x in self.map if x.moonphase()==0 and (not self.map.prev(x) or self.map.prev(x).moonphase()==2) ]
            '''if not l:
                print self.__class__.__name__ + ': nothing to compute'
                return
            if any([x.name==proposed_name for x in l]):
                n = [x for x in l if x.name == proposed_name][0]
            else:'''
                n = l[0]

            for vname in [x for x in vars(self) if x in shared.INHERITABLE_ATTR_LIST and getattr(self,x,None) and not getattr(n,x,None)]:
                setattr(n,x,getattr(self,x))
            n.prev = self.map.prev(n)
            n.compute()

        elif getattr(self, 'property', None):
            if not getattr(self, 'path', None):
                self.path = raw_input('Provide path for this node: \n %s \n >:' %str(self))    # counterpart implemented in sigmajs
            if not getattr(self, 'gen', None):
                self.gen = engine.Gen(self.phase + ', ' + self.property, self.cell)
            if not getattr(self, self.gen.getkw('engine'), None):
                engine_class = getattr(engine, self.gen.getkw('engine').title())
                engine_ = engine_class(self.gen, self.cell, self.path, self.prev)
                setattr(self, self.gen.getkw('engine'), engine_)
            getattr(self, self.gen.getkw('engine')).compute()

        else:
            raise shared.CustomError(self.__class___.__name__ + ': compute: Node %s is not computable.' %self.name)

    def delete(self):
            engine_name = self.gen.getkw('engine')
            engine_ = getattr(self,self.gen.getkw('engine'),None)
            if getattr(self,self.gen.getkw('engine'),None):
                getattr(self,self.gen.getkw('engine'),None).delete()
            for node in shared.NODES['master'].map.traverse():
                node.map.pop(self)
    '''

