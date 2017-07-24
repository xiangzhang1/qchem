# master

graph:
qd_base -> qd_facet
qd_base -> 2
qd_base ->3

## qd_base

graph:
bulk -> qd

### bulk

phase:
qd

cell:
poscar
1
1 0 0
0 1 0
0 0 1
H
1
D
0 0 0

map:
opt -> static

#### opt

property: 
engine=vasp, spin=fm

#### static

### qd

## qd_facet

## 2

## 3
