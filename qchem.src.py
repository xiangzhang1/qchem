import pickle
import re
import os
import engine
from shared import ELEMENTS, NODES


# ==================================================

# Node

def Import(text):
    l = re.split('^#+\s*', text, flags=re.MULTILINE) ; l.pop(0)

    while l:
        print 'Import: parsing %s' %(l[-1].splitlines()[0] if l[-1].splitlines() else '')
        n = Node(l.pop())
        '''if n.name in NODES:
            raise KeyError('Node name %s is in NODES.' %n.name)'''
        NODES[n.name] = n

def Dump():
    with open(os.path.dirname(os.path.abspath(__file__)+'/qchem.dump'),'w') as dumpfile:
        pickle.dump(NODES, dumpfile, protocol=pickle.HIGHEST_PROTOCOL)

def Load():
    with open(os.path.dirname(os.path.abspath(__file__)+'/qchem.dump'),'r') as dumpfile:
        pickle.load(dumpfile)



class Node(object):

    def __init__(self, text):   # parses 1 node at a time. searches in NODES

        namevalpairs = text.split('\n\n')

        # node.name
        '''if len(namevalpairs[0].splitlines()) != 1:
            raise SyntaxError('Section header format is name [: property]. Your header is:\n %s' %namevalpairs[0])'''
        titleline = namevalpairs.pop(0).splitlines()[0]
        l = [x.strip() for x in titleline.split(':')]
        self.name = l[0]
        if len(l) == 2: self.property = l[1]
        
        # node.__dict__
        
        while namevalpairs:
            namevalpair = namevalpairs.pop(0)
            if not namevalpair.rstrip():
                continue
            name = namevalpair.split('\n')[0].strip(': ')
            if name not in ['name','phase','cell','property','map']:    # This may NEED to be rather frequently UPDATED!
                continue
            value = namevalpair.split('\n',1)[1]
            if getattr(engine, name.title(), None):
                value = getattr(engine, name.title())(value)
            setattr(self, name, value)


    def edit(self, text):   
        # counterpart: str(self).
        # edit phase, cell, property, map at this level.
        for x in self._map:
            NODES[x.name] = x
        Import(text)
        '''if self.name in NODES:
            new_node = NODES[self.name]
        else:
            raise SyntaxError('Node with name %s is not defined in %s' %(self.name, filename))
        for varname in [x for x in vars(self) if getattr(new_node, x, None)]:
            setattr(self, varname, getattr(new_node, varname))'''


    def moonphase(self):
        if getattr(self, 'map', None):
            return min([x.moonphase() for x in self.map])
        elif getattr(self, 'gen', None) and getattr(self, self.gen.getkw('engine'), None):
            return getattr(self, self.gen.getkw('engine')).moonphase()
        elif getattr(self, 'property', None):
            return 0
        else:
            return 2
    
            
    '''def __str__(self):
        result = '# ' + self.name + '\n\n'
        for varname in vars(self):
            result += varname + ':\n' + str(vars(self)[varname]) + '\n\n'
        result += '\n\n'
    '''


    def compute(self, proposed_name=None):

        if getattr(self, 'map', None):

            l = [x for x in self.map if x.moonphase()==0] + [x for x in self._map if x.moonphase()==0 and self.prev(x).moonphase()==2]
            '''if not l:
                print self.__class__.__name__ + ': nothing to compute'
                return
            if any([x.name==proposed_name for x in l]):
                n = [x for x in l if x.name == proposed_name][0]
            else:'''
                n = l[0]

            for vname in [x for x in vars(self) if x not in ['name','map','path','property'] and getattr(self,x,None) and not getattr(n,x,None)]:
                setattr(n,x,getattr(self,x))
            n.prev = self.map.prev(n)
            n.compute()

        elif getattr(self, 'property', None):
            if not getattr(self, 'path', None):
                self.path = raw_input('Provide path for this node: \n %s \n >:' %str(self))
            if not getattr(self, 'gen', None):
                self.gen = engine.Gen(self.phase + ' ' + self.property, self.cell)
            if not getattr(self, self.gen.getkw('engine'), None):
                engine_class = getattr(engine, self.gen.getkw('engine').title())
                engine_ = engine_class(self.gen, self.cell, self.path, self.prev)
                setattr(self, self.gen.getkw('engine'), engine_)
            getattr(self, self.gen.getkw('engine')).compute()

        else:
            raise ValueError('Node %s is not computable.' %self.name)

        
    '''def delete(self):
            engine_name = self.gen.getkw('engine')
            engine_ = getattr(self,self.gen.getkw('engine'),None)
            if getattr(self,self.gen.getkw('engine'),None):
                getattr(self,self.gen.getkw('engine'),None).delete()
            for node in NODES['master'].map.traverse():
                node.map.pop(self)
    '''




