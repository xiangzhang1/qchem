# master

map:
qd_base -> qd_facet
qd_base -> 2
qd_base -> 3
qd_base -> 4

## qd_base

map:
bulk -> qd

### qd

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

#### opt : engine=vasp, spin=fm

#### static

## qd_facet

## 2

## 3
