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
import shutil
from pprint import pprint
from functools import wraps
import paramiko
import time


# INDEX
# =====
# DEBUG option, SCRIPT_DIR
# NODES, Fragile lists
# ELEMENTS
# CustomError
# MySFTPClient
# @memoize
# @moonphase_wrap

# 
# ===========================================================================

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) 

DEBUG = 1   
#DEBUG = 0    

# Nodes
# ===========================================================================

NODES = {}

# Fragile lists:

#  all non-sigma-bullshit attributes, in the order printed (sigma.clickNode)
#  - gui.combine.json
#  - sigma.clickNode (catch: map is not printed in sigma)
ALL_ATTR_LIST = ['property','phase','cell','comment','path','name','gen','vasp','electron','map']   #important as it is, we shall not print 'prev'

#  input attributes, not gen etc.
#  - qchem.Node.__init__
#  - sigma.edit_vars_addfield
READABLE_ATTR_LIST = ['name','phase','cell','property','map','comment','path']

#  for the inherit feature. 
#  - qchem.Node.compute
INHERITABLE_ATTR_LIST = ['phase','cell']


# UI-specific behavior

## gui.traverse_json: fuzzy match jam recipe

## gui.*beautify* recipe: name -> id

## gui|sigma . palette
COLOR_PALETTE = {-1: '#a94442', 0: '#000000',  1: '#8a6d3b', 2: '#3c763d', -2: '#000000'}   # text-*



# Element, ELEMENTS
# ===========================================================================

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

    def __init__(self, number, symbol, name, **kwargs):
        self.number = number
        self.symbol = symbol
        self.name = name
        self.electrons = number
        self.protons = number
        self.__dict__.update(kwargs)

    def __str__(self):
        # return self.name
        return repr(self)

    def __repr__(self):
        ionenergy = []
        for i, j in enumerate(self.ionenergy):
            if i and (i % 5 == 0):
                ionenergy.append("\n" + " " * 15)
            ionenergy.append("%s, " % j)
        ionenergy = "".join(ionenergy)

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
        # """Return electron configuration as dict."""
        adict = {}
        if self.eleconfig.startswith('['):
            base = self.eleconfig.split(' ', 1)[0][1:-1]
            adict.update(ELEMENTS[base].eleconfig_dict)
        for e in self.eleconfig.split()[bool(adict):]:
            adict[(int(e[0]), e[1])] = int(e[2:]) if len(e) > 2 else 1
        return adict

    @lazyattr
    def eleshells(self):
        # """Return number of electrons in shell as tuple."""
        eleshells = [0, 0, 0, 0, 0, 0, 0]
        for key, val in self.eleconfig_dict.items():
            eleshells[key[0] - 1] += val
        return tuple(ele for ele in eleshells if ele)

    def validate(self):
        # """Check consistency of data. Raise Error on failure."""
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
    # """Ordered dict of Elements with lookup by number, symbol, and name."""
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
        infile = open('shared.element.conf','r')
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
    # """Return string, word wrapped at linelen."""
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
ELEMENTS.import_()


# ===========================================================================

# Exceptions

class CustomError(Exception):
    pass


# ===========================================================================

# MySFTPClient: A wrapper for paramiko

import paramiko
import os

class MySFTPClient(paramiko.SFTPClient):
    def put_dir(self, source, target):
        # Uploads the contents of the source directory to the target path. The
        #    target directory needs to exists. All subdirectories in source are 
        #    created under target.
        for item in os.listdir(source):
            if os.path.isfile(os.path.join(source, item)):
                self.put(os.path.join(source, item), '%s/%s' % (target, item))
            else:
                self.mkdir('%s/%s' % (target, item), ignore_existing=True)
                self.put_dir(os.path.join(source, item), '%s/%s' % (target, item))

    def mkdir(self, path, mode=511, ignore_existing=False):
        # Augments mkdir by adding an option to not fail if the folder exists  '''
        try:
            super(MySFTPClient, self).mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise


# ===========================================================================

# @moonphase_wrap

def moonphase_wrap(func):
    moonphases = {}     # moonphases[self] = [timestamp, result]
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        if self in moonphases and time.time()-moonphases[self][0] < 20:
            if DEBUG == 2:  print 'moonphase: used cached value'
            return moonphases[self][1]
        elif getattr(self,'path',None) and os.path.isfile(self.path + '/.moonphase'):
            with open(self.path + '/.moonphase') as f:
                l = f.read().splitlines(0)
                if int(l[0]) > -2 and int(l[0]) < 3:
                    return int(l[0])
        else:
            result = func(self, *args, **kwargs)
            moonphases[self] = [time.time(), result]
            return result
    return wrapped



