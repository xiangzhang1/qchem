import engine
from shared import ELEMENTS, NODES
import re



def Import(filename):

    with open(filename, 'r') as f:
        l = re.split('(^#+\s+)', f.read(), flags=re.MULTILINE) ; l.pop(0)

    while l:
        n = Node(l.pop())
        '''if n.name in NODES:
            raise KeyError('Node name %s is in NODES.' %n.name)'''
        NODES[n.name] = n



class Node(object):


    def __init__(self, text):

        chunks = text.split('\n\n')

        # node.name
        '''if len(chunks[0].splitlines()) != 1:
            raise SyntaxError('Section header format is name [: property]. Your header is:\n %s' %chunks[0])'''
        line = chunks.pop(0).splitlines()[0]
        l = [x.strip() for x in line.split(':')]
        self.name = l[0]
        if len(l) == 2:
            self.property =line[1]
        
        # node.__dict__
        while chunks:
            chunk = chunks.pop(0).splitlines()
            attr_name = chunk.pop(0).split(':')[0].strip()
            chunk = '\n'.join(chunk)
            attr_value = chunk
            if getattr(engine, attr_name.title(), None):
                attr_value = getattr(engine, attr_name.title())(attr_value)
            setattr(self, attr_name, attr_value)


    def edit(self, filename):
        '''if not os.path.isfile(filename):
            with open(filename, 'w') as f:'''
                f.write(str(self))
        '''else:
            for x in self._map:
                NODES[x.name] = x'''
            Import(filename)
            '''if self.name in NODES:
                new_node = NODES[self.name]
            else:
                raise SyntaxError('Node with name %s is not defined in %s' %(self.name, filename))
            for varname in [x for x in vars(self) if getattr(new_node, x, None):
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
        result = self.name + '\n\n\n'
        for varname in [x for x in vars(self) if x in ['name','phase','cell','property','map']]:
            result += varname + ':\n' + str(vars(self)[varname]) + '\n\n'
        result += '\n\n\n'
    '''


    def compute(self, proposed_name=None):

        if getattr(self, 'map', None):

            l = [x for x in self.map if x.moonphase()==0] + [x for x in self._map if x.moonphase()==0 and self.prev(x).moonphase()==2]
            '''if not l:
                print self.__class__.__name__ + ': nothing to compute'
                return
            if any([x.name==proposed_name for x in l]):
                n = next([x for x in l if x.name == proposed_name])
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
            for node in NODES['master'].traverse():
                node.map.pop(self)

    def traverse(self):
        return set([x for x in self.map]) + set().union( *(x.traverse() for x in self.map) )
    '''
