import engine
import shared



def Import(filename):

    with open(filename, 'r') as f:
        l = f.read().split('\n\n\n')

    while l:
        n = Node(l.pop(0))
        if n.name in NODES:
            raise KeyError('Node name %s is in NODES.' %n.name)
        NODES[n.name] = n



class Node(object):


    def __init__(self, text):

        chunks = text.split('\n\n')

        # node.name
        if len(chunks[0].splitlines()) != 1:
            raise SyntaxError('First part of a chunk is name [: property] \n')
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


