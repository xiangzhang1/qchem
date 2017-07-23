import engine
from shared import ELEMENTS, NODES



def Import(filename):

    with open(filename, 'r') as f:
        l = f.read().split('\n\n\n')

    while l:
        n = Node(l.pop(0))
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
            with open(filename, 'w') as f:
                f.write(str(self))
        else:
            with open(filename, 'r') as f:
                text = f.read()'''
            for x in self._map:
                NODES[x.name] = x
            new_node = Node(text)
            '''for varname in vars(self):
                set(self, varname, getattr(new_node, varname, None))'''


    '''def moonphase(self):    #0=pre,1=peri,2=post
        if getattr(self, 'map', None):
            return min([x.moonphase() for x in self.map])
        elif getattr(self, 'gen', None) and getattr(self, self.gen.getkw('engine'), None):
            return getattr(self, self.gen.getkw('engine')).moonphase()
        else:
            return 0
    
            
    def __str__(self):
        result = self.name + '\n\n\n'
        for varname in vars(self):
            result += varname + ':\n' + str(vars(self)[varname]) + '\n\n'
        result += '\n\n\n'
    '''


    def compute(self, proposed=None):

        if getattr(self, 'map', None):
            l = self.map.compute()
            n = next([x for x in self.map if x.name == proposed]) if proposed else l[0]
            n.phase, n.cell = self.phase, self.cell
            n.compute()

        elif not getattr(self, 'property', None):
            raise ValueError('Node %s is blank. ' %self.name)

        else:
            if not getattr(self, 'path', None):
                self.path = raw_input('Provide path for this node: \n %s \n >:' %str(self))
            if not getattr(self, 'gen', None):
                self.gen = engine.Gen(self.phase + ' ' + self.property)
            if not getattr(self, self.gen.getkw('engine'), None):
                engine_class = getattr(engine, self.gen.getkw('engine').title())
                engine_ = engine_class(self.gen, self.cell, self.path)
                setattr(self, self.gen.getkw('engine'), engine_)
            getattr(self, self.gen.getkw('engine')).compute()


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
