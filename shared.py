#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import itertools
import sys
import numpy as np
import math
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
from StringIO import StringIO
from sklearn.preprocessing import LabelBinarizer


# INDEX
# =====
# DEBUG option, SCRIPT_DIR, HOME_DIR
# Fragile lists
# elements
# CustomError, DeferError, [LookupError]
# MySFTPClient, sshs
# @MWT
# @moonphase_wrap
# wraparound error constant
# @log_wrap
# bcolors (different from Fragile lists)
# @debug_wrap
# pipeline-friendly LabelBinary
# euler2mat

#
# ===========================================================================

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
HOME_DIR = os.path.expanduser(os.path.expanduser("~")) + '/'
SCRATCH_DIR = SCRIPT_DIR + '/.scr'

DEBUG = 0

# Pickle-based variable persistency
# ===========================================================================
# - save to data directory
# - automatic version control
# shorter code
def save(obj, middlename):      # Note! Where defined, above which obj pickled.
    filepath = shared.SCRIPT_DIR + '/data/dynamic.%s.pickle.'%(middlename) + time.strftime('%Y%m%d%H%M%S')
    with open(filepath,'wb') as dumpfile:
        pickle.dump(obj, dumpfile, protocol=pickle.HIGHEST_PROTOCOL)

def load(middlename, datetime=None):
    filepath = shared.SCRIPT_DIR + '/data/' + sorted([x for x in os.listdir(shared.SCRIPT_DIR + '/data/') if x.startswith('dynamic.%s.pickle.%s'%(middlename, datetime if datetime else ''))])[-1]
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Fragile lists:
# ===========================================================================

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
INHERITABLE_ATTR_LIST = ['phase']   # , 'cell': cell is inherited from prev
PREV_INHERITABLE_ATTR_LIST = ['cell']


# UI-specific behavior

## gui.traverse_json: fuzzy match jam recipe

## gui.*beautify* recipe: name -> id

## gui|sigma . palette
COLOR_PALETTE = {-1: '#a94442', 0: '#000000',  1: '#8a6d3b', 2: '#3c763d', -2: '#000000'}   # text-*



# Element, elements
# ===========================================================================

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
            adict.update(elements[base].eleconfig_dict)
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
        self.orbitals = 's p_y p_z p_x d_xy d_yz d_z2 d_xz dx2-y2'.split()   # user added
        self.spins = {1: [''], 2: [u'\u2191', u'\u2193'], 3:['_x','_y','_z'], 4: ['','_x','_y','_z']}   # user added
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
        infile = open(SCRIPT_DIR + 'conf/shared.element.conf','r')
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

elements = ElementsDict()
elements.import_()


# ===========================================================================

# Exceptions

class CustomError(Exception):
    pass


class DeferError(Exception):   # DeferError name seems to conflict
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

sshs = {}
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
                if DEBUG>=2:    print "cache"
                if (time.time() - v[1]) > self.timeout:
                    raise KeyError
            except KeyError:
                if DEBUG>=2:    print "new"
                v = self.cache[key] = f(*args,**kwargs),time.time()
            return v[0]
        func.func_name = f.func_name

        return func

# ===========================================================================

# @moonphase_wrap

def moonphase_wrap(func):
    @MWT(timeout=60)
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

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        sys.stdout.flush()

def log_wrap(func):
    @debug_wrap
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        # change stdout to dual-pipe
        sys.stdout = Logger()
        # pipe to both stdout and sys.stdout.log->self.log
        print '\n'
        print '*' * 30 + ' ' + self.__class__.__name__ + ' @ ' + os.getcwd() + ' ' + '*' * 30
        result = func(self, *args, **kwargs)    # the important part
        print '*' * len('*' * 30 + ' ' + self.__class__.__name__ + ' @ ' + os.getcwd() + ' ' + '*' * 30 + '\n\n')
        self.log = sys.stdout.log.getvalue()    # sys.stdout now is the Logger object.
        # change stdout back
        sys.stdout = sys.__stdout__
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
        if DEBUG >= 2:
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



# ===========================================================================

class LabelBinarizerPipelineFriendly(LabelBinarizer):
     def fit(self, X, y=None):
         """this would allow us to fit the model based on the X input."""
         super(LabelBinarizerPipelineFriendly, self).fit(X)
     def transform(self, X, y=None):
         return super(LabelBinarizerPipelineFriendly, self).transform(X)

     def fit_transform(self, X, y=None):
         return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)


# ===========================================================================

def euler2mat(z=0, y=0, x=0):
    ''' Return matrix for rotations around z, y and x axes

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles

    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True

    The output rotation matrix is equal to the composition of the
    individual rotations

    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True

    You can specify rotations by named arguments

    >>> np.all(M3 == euler2mat(x=xrot))
    True

    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.

    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)

    Rotations are counter-clockwise.

    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True

    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)
