#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import re
import numpy as np
import scipy

class Poscar(object):   # Parses poscar.
    def __init__(self):
        f=open("POSCAR","r")
        self.lines=f.readlines()
        self.cell=[self.lines[i].split() for i in range(2,5)]
        self.base=[self.lines[i].split()[0:3] for i in range(8,len(self.lines))]
        self.elements = self.lines[5].split()
        self.nelements = len(self.elements)
        self.atomcounts = np.int_(self.lines[6].split())
        self.natoms = sum(self.atomcounts)
        self.cell=np.float64(self.cell)
        self.base=np.float64(self.base)
        f.close()
        with open('POSCAR','r') as infile:
            self.lines = infile.read().splitlines()
            self.name = self.lines[0]
            self.cell = np.float_([ line.split() for line in lines[2:5] ]) * float(lines[1])
            if 
            self.elements = dict( zip(lines[5].split(), map(int,lines[6])) )
            if not lines[7].startswith('D'):
                print 'pre.poscar error: unsupported format. Only direct coordinates are supported. Explanation: who the heck uses Cartesian?'

