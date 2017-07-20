#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import itertools
import sys
import numpy as np
import scipy
import os
import re
from shutil import copyfile



# Element, ELEMENTS
# =========================================================================


__version__ = '2015.01.29'
__docformat__ = 'restructuredtext en'

class lazyattr(object):
    # Lazy object attribute whose value is computed on first access."""
    __slots__ = ['func']

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        result = self.func(instance)
        if result is NotImplemented:
            return getattr(super(owner, instance), self.func.__name__)
        setattr(instance, self.func.__name__, result)
        return result


class Element(object):
    """Chemical element.
    """
    def __init__(self, number, symbol, name, **kwargs):
        self.number = number
        self.symbol = symbol
        self.name = name
        self.electrons = number
        self.protons = number
        self.__dict__.update(kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        ionenergy = []
        for i, j in enumerate(self.ionenergy):
            if i and (i % 5 == 0):
                ionenergy.append("\n" + " " * 15)
            ionenergy.append("%s, " % j)
        ionenergy = "".join(ionenergy)

        description = word_wrap(self.description, linelen=66, indent=0,
                                joinstr=""" "\n        \"""")
        description = """    e['%s'].description = (\n        "%s\")""" % (
            self.symbol, description)
        # return description

        result = [
            "Element(\n    %i, '%s', '%s'" % (
                self.number, self.symbol, self.name),
            "group=%s, period=%s, block='%s', series=%i" % (
                self.group, self.period, self.block, self.series),
            "mass=%s, eleneg=%s, eleaffin=%s" % (
                self.mass, self.eleneg, self.eleaffin),
            "covrad=%s, atmrad=%s, vdwrad=%s" % (
                self.covrad, self.atmrad, self.vdwrad),
            "tboil=%s, tmelt=%s, density=%s" % (
                self.tboil, self.tmelt, self.density),
            "eleconfig='%s'" % self.eleconfig,
            "oxistates='%s'" % self.oxistates,
            "ionenergy=(%s)" % ionenergy
        ]
        return ",\n    ".join(result)

    @lazyattr
    def neutrons(self):
        return self.nominalmass - self.protons

    @lazyattr
    def eleconfig_dict(self):
        """Return electron configuration as dict."""
        adict = {}
        if self.eleconfig.startswith('['):
            base = self.eleconfig.split(' ', 1)[0][1:-1]
            adict.update(ELEMENTS[base].eleconfig_dict)
        for e in self.eleconfig.split()[bool(adict):]:
            adict[(int(e[0]), e[1])] = int(e[2:]) if len(e) > 2 else 1
        return adict

    @lazyattr
    def eleshells(self):
        """Return number of electrons in shell as tuple."""
        eleshells = [0, 0, 0, 0, 0, 0, 0]
        for key, val in self.eleconfig_dict.items():
            eleshells[key[0] - 1] += val
        return tuple(ele for ele in eleshells if ele)

    @lazyattr
    def description(self):
        """Return text description of element."""
        return _descriptions(self.symbol)

    def validate(self):
        """Check consistency of data. Raise Error on failure."""
        assert self.period in PERIODS
        assert self.group in GROUPS
        assert self.block in BLOCKS
        assert self.series in SERIES

        if self.number != self.protons:
            raise ValueError(
                "%s - atomic number must equal proton number" % self.symbol)
        if self.protons != sum(self.eleshells):
            raise ValueError(
                "%s - number of protons must equal electrons" % self.symbol)

class ElementsDict(object):
    """Ordered dict of Elements with lookup by number, symbol, and name."""
    def __init__(self, *elements):
        self._list = []
        self._dict = {}
        for element in elements:
            if element.number > len(self._list) + 1:
                raise ValueError("Elements must be added in order")
            if element.number <= len(self._list):
                self._list[element.number - 1] = element
            else:
                self._list.append(element)
            self._dict[element.number] = element
            self._dict[element.symbol] = element
            self._dict[element.name] = element

    def add(self,*elements):
        for element in elements:
            if element.number > len(self._list) + 1:
                raise ValueError("Elements must be added in order")
            if element.number <= len(self._list):
                self._list[element.number - 1] = element
            else:
                self._list.append(element)
            self._dict[element.number] = element
            self._dict[element.symbol] = element
            self._dict[element.name] = element
    
    def import_(self):
        infile = open('phase.element.conf','r')
        lines = [[field.strip() for field in line.split(':')] for line in infile.read().splitlines()]
        header = lines[0]
        for line_idx, line in enumerate(lines):
            if line_idx == 0:
                continue
            exp = 'Element('
            for idx in range(0,len(line)):
                exp += header[idx] + '=' + line[idx]
                if idx != len(line)-1:
                    exp += ','
            exp += ')'
            self.add( eval(exp) )
        
    def export_merge(self):
        # BACKUP CONF FILE!
        # header
        outfile = open('phase.element.conf','w')
        outfile.write('%10s:%10s:%15s:%10s:%10s:%10s:%10s:%16s:%10s:%10s:%10s:%10s:%10s:%10s:%10s:%10s:%30s:%30s:%180s:%10s:%10s:%170s:%10s:%15s:%150s:%15s:%10s\n' % (
                'number', 'symbol', 'name', 'group', 'period', 'block', 'series', 'mass', 'eleneg', 'eleaffin', 
                'covrad', 'atmrad', 'vdwrad', 'tboil', 'tmelt', 'density', 'eleconfig', 'oxistates', 'ionenergy', 'ldauu', 'ldauj',
                'ldaucomment', 'magmom', 'magmomcomment', 'pot', 'pot_extension','pot_zval')) #EDIT HEADER HERE
        # read and write
        infile = open('ZVALdatabase','r')  #EDIT INFILE HERE
        lines = [[field.strip() for field in line.split(':')] for line in infile.read().splitlines()]
        for e in self:
            matchingline = [line for line in lines if line[0] == e.symbol]
            if len(matchingline) == 0:
                pot_zval = None
            else:
                pot_zval = matchingline[0][1] #EDIT TO_OUTPUT VARS HERE
            # original elements formatting
            ionenergy = []
            for i, j in enumerate(e.ionenergy):
                ionenergy.append("%s, " % j)
            ionenergy = "".join(ionenergy)
            # write
            outfile.write('%10i:%10s:%15s:%10s:%10s:%10s:%10i:%16s:%10s:%10s:%10s:%10s:%10s:%10s:%10s:%10s:%30s:%30s:%180s:%10s:%10s:%170s:%10s:%15s:%150s:%15s:%10s\n' % ( 
                e.number, '\''+e.symbol+'\'', '\''+e.name+'\'', e.group, e.period, '\''+e.block+'\'', e.series,
                e.mass, e.eleneg, e.eleaffin, e.covrad, e.atmrad, e.vdwrad, e.tboil, e.tmelt, e.density,
                '\''+e.eleconfig+'\'', '\''+e.oxistates+'\'', '('+ionenergy+')', e.ldauu, e.ldauj, '\''+e.ldaucomment+'\'', 
                e.magmom, '\''+e.magmomcomment + '\'', e.pot, '\''+e.pot_extension+'\'', pot_zval)) 
            #EDIT OUTPUT HERE. NOTE: STRING, WHETHER ORIGINAL OR BEING IMPORTED, MUST BE SURROUNDED BY QUOTES! 
        outfile.close()
        infile.close()
        
    def __str__(self):
        return "[%s]" % ", ".join(ele.symbol for ele in self._list)

    def __contains__(self, item):
        return item in self._dict

    def __iter__(self):
        return iter(self._list)

    def __len__(self):
        return len(self._list)

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except KeyError:
            try:
                start, stop, step = key.indices(len(self._list))
                return self._list[slice(start - 1, stop - 1, step)]
            except:
                raise KeyError


def word_wrap(text, linelen=80, indent=0, joinstr="\n"):
    """Return string, word wrapped at linelen."""
    if len(text) < linelen:
        return text
    result = []
    line = []
    llen = -indent
    for word in text.split():
        llen += len(word) + 1
        if llen < linelen:
            line.append(word)
        else:
            result.append(" ".join(line))
            line = [word]
            llen = len(word)
    if line:
        result.append(" ".join(line))
    return joinstr.join(result)

ELEMENTS = ElementsDict()
#ELEMENTS.import_()
#ELEMENTS.export_merge()
ELEMENTS.import_()


# =========================================================================== 

import engine

KETS = {}

class Ket(object):

    def __init__(self, phase, cell, property_wanted, path,uid):
        self.phase, self.cell, self.property_wanted, self.path = phase, cell, property_wanted, path
        self.uid = uid if uid else os.path.basename(os.path.normpath(self.path))
        # dynamic composition
        for p in self.phase:
            constructor = globals()[p]
            setattr(self, p, constructor(self))
        # add to KETS
        if uid in KETS:
            raise SyntaxError('uid %s already in KETS' %uid)
        else:
            KETS[uid] = self
    
    def moonphase(self):
        if len(self.property_wanted.nodes) != 1:    # workflow, phantom
            return self.property_wanted.moonphase()
        else:   # copy
            engine_ = getattr(self,self.gen.getkw('engine'),None)
            if engine_:
                return engine_.moonphase()
            else:
                return 0

    def compute(self):
        # phantom or workflow
        if len(self.property_wanted.nodes) != 1:
            next_uid = self.property_wanted.compute()
            prev_uid = self.property_wanted.prev(next_uid)
            uid = node.keys()[0]
            path = node[uid][1]
            if uid not in KETS:
                KETS[uid] = Ket(self.phase, self.cell, self.property_wanted.subset([next_uid]), self.property_wanted.nodes[next_uid][1], uid)
                KETS[uid].compute()
            else:
                KETS[uid].compute()
        # copy
        else:
            self.gen = engine.Gen(' '.join(self.phase) + ' ' + self.property_wanted.nodes.values()[0][0], self.cell)
            name_ = self.gen.getkw('engine')
            engine_ = getattr(self,self.gen.getkw('engine'),None)
            if not engine_:
                class_ = getattr(engine, name_.title())
                engine_ = class_(self.gen, self.cell, self.path, self.uid)
                setattr(self, name_, engine_)
                getattr(self, name_).compute()

    def edit(self):
        input_ = raw_input('Enter a filename to read/write, or just call ket.property_wanted.read_(): ')
        if os.path.isfile(input_):
            self.property_wanted.read_(if_=input_)
        else:
            self.property_wanted.write_(of_=input_)

    def delete(self):
            name_ = self.gen.getkw('engine')
            engine_ = getattr(self,self.gen.getkw('engine'),None)
            if engine_:
                engine_.delete()
            KETS.pop(self.uid)

    def consider(self,uid):
        return KETS[uid]
        

# =========================================================================== 

class Qd(object):
    pass


