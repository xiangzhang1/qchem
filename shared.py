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
import traceback, sys, code


# INDEX
# =====
# DEBUG option, SCRIPT_DIR, HOME_DIR
# NODES, Fragile lists
# ELEMENTS
# CustomError, DeferError
# MySFTPClient
# @MWT
# @moonphase_wrap
# wraparound error constant
# @log_wrap
# bcolors (different from Fragile lists)
# @debug_wrap

#
# ===========================================================================

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
HOME_DIR = os.path.expanduser(os.path.expanduser("~"))

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
        self.orbitals = 's p_y p_z p_x d_xy d_yz d_z2 d_xz d_x^2-y^2'.split()   # user added
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
        infile = open('conf/shared.element.conf','r')
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

class DeferError(Exception):
    pass

# ===========================================================================

# MySFTPClient: A wrapper for paramiko

class MySFTPClient(paramiko.SFTPClient):
    '''extension of paramiko SFTPClient'''

    def put_dir(self, source, target):
        ''' Uploads the contents of the source directory to the target path. The
            target directory needs to exists. All subdirectories in source are
            created under target.'''
        for item in os.listdir(source):
            if os.path.isfile(os.path.join(source, item)):
                self.put(os.path.join(source, item), '%s/%s' % (target, item))
            else:
                self.mkdir('%s/%s' % (target, item), ignore_existing=True)
                self.put_dir(os.path.join(source, item), '%s/%s' % (target, item))

    def mkdir(self, path, mode=511, ignore_existing=False):
        '''Augments mkdir by adding an option to not fail if the folder exists'''
        try:
            super(MySFTPClient, self).mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise

# ===========================================================================

# MWT: Memorize With Timeout

class MWT(object):
    """Memoize With Timeout"""
    _caches = {}
    _timeouts = {}

    def __init__(self,timeout=86400):
        self.timeout = timeout

    def collect(self):
        """Clear cache of results which have timed out"""
        for func in self._caches:
            cache = {}
            for key in self._caches[func]:
                if (time.time() - self._caches[func][key][1]) < self._timeouts[func]:
                    cache[key] = self._caches[func][key]
            self._caches[func] = cache

    def __call__(self, f):
        self.cache = self._caches[f] = {}
        self._timeouts[f] = self.timeout

        def func(*args, **kwargs):
            kw = kwargs.items()
            kw.sort()
            key = (args, tuple(kw))
            try:
                v = self.cache[key]
                if DEBUG==2:    print "cache"
                if (time.time() - v[1]) > self.timeout:
                    raise KeyError
            except KeyError:
                if DEBUG==2:    print "new"
                v = self.cache[key] = f(*args,**kwargs),time.time()
            return v[0]
        func.func_name = f.func_name

        return func

# ===========================================================================

# @moonphase_wrap

def moonphase_wrap(func):
    @MWT(timeout=10)
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        if getattr(self,'path',None) and os.path.isfile(self.path + '/.moonphase'):
            with open(self.path + '/.moonphase') as f:
                l = f.read().splitlines(0)
                if int(l[0]) > -2 and int(l[0]) < 3:
                    return int(l[0])
        else:
            result = func(self, *args, **kwargs)
            return result
    return wrapped


# ===========================================================================

# @log_wrap

def log_wrap(func):
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        self.log = '\n\n'
        self.log += '*' * 30 + ' ' + self.__class__.__name__ + ' @ ' + os.getcwd() + ' ' + '*' * 30 + '\n'
        result = func(self, *args, **kwargs)    # the important part
        self.log += '*' * len('*' * 30 + ' ' + self.__class__.__name__ + ' @ ' + os.getcwd() + ' ' + '*' * 30 + '\n\n')   #matching length
        print self.log
        return result
    return wrapped

# ===========================================================================

# @bcolors

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ===========================================================================

# debug_wrap

def debug_wrap(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if DEBUG > 0:
            try:
                return func(*args, **kwargs)    # the important part
            except:
                print bcolors.FAIL + 'error invoked, debug starting at frame of error:\n' + bcolors.ENDC
                type, value, tb = sys.exc_info()
                traceback.print_exc()
                last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
                frame = last_frame().tb_frame
                ns = dict(frame.f_globals)
                ns.update(frame.f_locals)
                code.interact(local=ns)
                print bcolors.FAIL + 'debug ended\n:' + bcolors.ENDC
                sys.exit()
        else:
            return func(*args, **kwargs)
    return wrapped
