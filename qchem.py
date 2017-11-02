#!/usr/bin/python
'''
This is the main file.
'''
import dill as pickle   # dill requires citation
import re
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import time
import shutil

from keras.models import load_model

import engine
import shared

# ==================================================

# Node

# Note: The Import function is becoming little used.
# def Import(text):
#
#     #:partial syntax check
#     if '#' not in text:
#         raise shared.CustomError('qchem.Import: bad syntax. Your text is {%s}.' %text)
#     #;
#     l = re.split('^#+\s*', text, flags=re.MULTILINE) ; l.pop(0)
#     l = ['# '+x for x in l]
#
#     while l:
#         # print 'Import: parsing %s' %(l[-1].splitlines()[0] if l[-1].splitlines() else '')
#         n = Node(l.pop())
#         #: name must not be in shared.NODES
#         if n.name in shared.NODES:
#             raise shared.CustomError(' Import: Node name %s is in already in shared.NODES.' %n.name)
#         #;
#         shared.NODES[n.name] = n

def _Dump(obj, middlename, is_keras=False):
    if is_keras:
        # keras-save the model
        filepath = shared.SCRIPT_DIR + '/data/shared.%s.model.keras.'%(middlename) + time.strftime('%Y%m%d%H%M%S')
        obj.model.save(filepath)
        tmp_model = obj.model
        delattr(obj, 'model')
        # regular save the rest
        _Dump(obj, middlename, is_keras=False)
        obj.model = tmp_model
    else:
        # dump
        filepath = shared.SCRIPT_DIR + '/data/shared.%s.pickle.'%(middlename) + time.strftime('%Y%m%d%H%M%S')
        with open(filepath,'wb') as dumpfile:
            pickle.dump(obj, dumpfile)

def _Load(middlename, datetime=None, is_keras=False):
    if not is_keras:
        # load
        filepath = shared.SCRIPT_DIR + '/data/' + sorted([x for x in os.listdir(shared.SCRIPT_DIR + '/data/') if x.startswith('shared.%s.pickle.%s'%(middlename, datetime if datetime else ''))])[-1]
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        # regular load the rest
        obj = _Load(middlename, datetime, is_keras=False)
        # keras-load the model
        filepath = shared.SCRIPT_DIR + '/data/' + sorted([x for x in os.listdir(shared.SCRIPT_DIR + '/data/') if x.startswith('shared.%s.model.keras.%s'%(middlename, datetime if datetime else ''))])[-1]
        obj.model = load_model(filepath)
        return obj


def Dump():
    # distinguish data source by prefix: shared.NODES.dump, shared.NODES.markdown, sigma.dump
    # distinguish datetime by postfix: 20170803094500
    if 'master' not in shared.NODES:
        raise shared.CustomError('Dump: NODES is empty. You really should not dump.')
    _Dump(shared.NODES, 'NODES', is_keras=False)
    _Dump(shared.ML_VASP_MEMORY, 'ML_VASP_MEMORY', is_keras=True)
    print 'Dump complete.'

def Load(datetime=None):
    shared.NODES = _Load('NODES', datetime=datetime, is_keras=False)
    shared.ML_VASP_MEMORY = _Load('ML_VASP_MEMORY', datetime=datetime, is_keras=True)
    print 'Load complete.'


class Node(object):

    def __init__(self, text='# newnode'):
        # parses 1 node at a time. searches in NODES.
        # default to a newnode

        namevalpairs = text.split('\n\n')

        # node.name
        #: partial grammar check
        if len(namevalpairs[0].splitlines()) != 1 or not namevalpairs[0].startswith('#'):
            raise shared.CustomError(self.__class__.__name__ +': __init__: Titleline format is #name:opt. Your titleline is: {%s}' %namevalpairs[0])
        #;
        titleline = namevalpairs.pop(0).rstrip().lstrip('# ')
        l = [x.strip() for x in titleline.split(':')]
        self.name = l[0]
        if len(l) == 2: self.property = l[1]

        # node.__dict__
        while namevalpairs:
            namevalpair = namevalpairs.pop(0)
            #: skip check
            if not namevalpair.rstrip():
                continue
            name = namevalpair.split('\n')[0].strip(': ')
            if name not in shared.READABLE_ATTR_LIST:
                continue
            #;
            value = namevalpair.split('\n',1)[1] if len(namevalpair.split('\n',1))>1 else ''
            if getattr(engine, name.title(), None):
                value = getattr(engine, name.title())(value)
            setattr(self, name, value)

        #: test gen if possible
        if getattr(self,'cell',None) and getattr(self,'phase',None) and getattr(self,'property',None):
           test_gen = engine.Gen(self)
        #;


    def reset(self):
        # reset moonphase = 1. remove all non-readable attributes.
        # remove engine
        if getattr(self, 'gen', None):
            engine_name = self.gen.getkw('engine')
            engine_ = getattr(self,self.gen.getkw('engine'),None)
            if getattr(self,self.gen.getkw('engine'),None):
                getattr(self,self.gen.getkw('engine'),None).delete()
        # remove all
        for varname in vars(self).keys():
            if varname not in shared.READABLE_ATTR_LIST:
                delattr(self, varname)
                print self.__class__.__name__ + '.reset: attribute {%s} deleted' %varname
        if getattr(self, 'path', None):
            if os.path.isdir(self.path):
                shutil.rmtree(self.path)
                print self.__class__.__name__ + '.reset: removed directory {%s}' %self.path


    @shared.moonphase_wrap
    def moonphase(self):
        if getattr(self, 'map', None):
            min_moonphase = min([x.moonphase() for x in self.map])
            max_moonphase = max([x.moonphase() for x in self.map])
            if min_moonphase > 0 or max_moonphase<1:
                return min_moonphase
            else:   # complex scenario. 1->0 counts as 1.
                prev = engine.Map().rlookup(node_list=[x], prev=True)
                if [x for x in self.map if x.moonphase()==0 and (not prev or prev.moonphase()==2) ]:
                    return 0
                else:
                    return 1
        elif getattr(self, 'gen', None) and getattr(self, self.gen.getkw('engine'), None):
            return getattr(self, self.gen.getkw('engine')).moonphase()
        elif getattr(self, 'property', None):
            return 0
        else:
            return -2

    def __str__(self):
        result = '# ' + self.name + '\n\n'
        for varname in vars(self):
            if varname != 'name':
                result += varname + ':\n' + str(vars(self)[varname]) + '\n\n'
        result += '\n\n'
        return result


    def compute(self, proposed_name=None):

        if shared.DEBUG >= 0:
            print 'computing node {%s}' %(self.default_path())

        if getattr(self, 'map', None):
            # next to calculate
            l = [x for x in self.map if x.moonphase()==0 and not [n for n in self.map if n.moonphase()!=2 and x in self.map[n]]]
            #: consider the proposed name
            if not l:
                raise shared.CustomError( self.__class__.__name__ + ': nothing to compute in parent node {%s]' %self.name )
            if proposed_name:
                tmp_l = [x for x in l if x.name == proposed_name]
                if not tmp_l:
                    raise shared.CustomError( self.__class__.__name__ + '.compute: cannot found proposed_name {%s} in map' %proposed_name)
                n = tmp_l[0]
            else:
            #;
                n = l[0]
            n.compute()

        elif getattr(self, 'property', None):
            # important inherits
            parent = engine.Map().rlookup(node_list=[self], parent=True)
            for vname in vars(parent):
                if vname in shared.INHERITABLE_ATTR_LIST and not getattr(self, vname, None):
                    setattr(self, vname, getattr(parent,vname))
            prev = engine.Map().rlookup(node_list=[self], prev=True)
            if prev:
                for vname in vars(prev):
                    if vname in shared.PREV_INHERITABLE_ATTR_LIST and not getattr(self, vname, None):
                        setattr(self, vname, getattr(prev,vname))
            if not getattr(self, 'cell', None) or not getattr(self, 'phase', None):
                raise shared.CustomError(self.__class__.__name__ + '.compute: cell or phase is missing. Either make sure parent has something you can inherit, or enter them.')
            if not getattr(self, 'path', None):
                if shared.DEBUG >=1:
                    self.path = raw_input('Provide path for node {%s}, empty to use default:' %self.name)    # counterpart implemented in sigmajs
                    if not self.path.strip():   self.path = self.default_path()
                else:   # silent mode
                    self.path = self.default_path()
                    if os.path.exists(self.path):
                        number = int(re.search(r'\d+$', self.path).group(0)) if re.search(r'\d+$', self.path) else 0
                        text = re.sub(r'\d+$', '', self.path)
                        while True:
                            if os.path.exists(text + str(number)):
                                number += 1
                            else:
                                self.path = text + str(number)
                                break
            if not getattr(self, 'gen', None):
                self.gen = engine.Gen(self)
            if not getattr(self, self.gen.getkw('engine'), None):
                engine_class = getattr(engine, self.gen.getkw('engine').title())
                engine_ = engine_class(self)
                setattr(self, self.gen.getkw('engine'), engine_)
            if shared.DEBUG >= 2:
                print self.__class__.__name__ + '.compute: computing engine %s' %(self.gen.getkw('engine'))
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

    def default_path(self):
        if engine.Map().lookup('master') == self:
            return '/home/xzhang1/run/master'
        else:
            parent_node = engine.Map().rlookup(node_list = [self], parent=True)     # if none found, an error would have been raised
            return parent_node.default_path() + '/' + re.sub(r"\s+", '_', self.name)
