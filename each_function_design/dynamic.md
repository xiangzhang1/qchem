State variables. Contain precious data. We don't want to lose them.

_load_(), _save_()
Generic load and dump

_nodes_
Dict of nodes. Usually just 'master', except when copy-and-pasting.

_MLS_
Dict of Machine Learning models. And their definitions as well.





VASP memory usage
======================
X: projector_real, projector_reciprocal, wavefunction, array_grid, natoms, volume, npar, ncore_total; lsorbit=0|1, isym=-1|0|1|2.
y: memory in byte
