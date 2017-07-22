#!/usr/bin/python
import os
import sys
import numpy as np
import scipy
import subprocess
import commands

from phase import ELEMENTS
of_ = open('ZVALdatabase','w')
for e in ELEMENTS:
    output = commands.getstatusoutput("pott "+e.symbol+" | grep ZVAL | awk '{print $6}'")
    print e.symbol, output[1]
    output = int(float(output[1])) if output[1].strip()!='' else None
    of_.write(e.symbol + ' : ' + str(output) + '\n' )
of_.close()
