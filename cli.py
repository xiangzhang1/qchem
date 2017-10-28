#!/usr/bin/python
'''Eventually, you should use CLI, not the pretty GUI.
This is a simple and stable wrapper.'''

# common libraries
import sys
import os
import shutil
import random
import string
import dill as pickle
import time
from pprint import pprint
import IPython
import numpy as np

from cStringIO import StringIO
from fuzzywuzzy import process


# qchem package
import qchem
import shared
import engine
from shared import ELEMENTS

print shared.bcolors.OKBLUE + 'Welcome. Libraries loaded.' + shared.bcolors.ENDC
IPython.embed(banner1='')
