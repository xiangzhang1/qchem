Grammar
-------

import PEP8

text __var__
- __key__ in __var__

__dos_interp__  __bands__[_idx_band_]  __nbands__  __idxs_band_neargap__

bands [_idx_spin_=0] [_idx_atom_==0] [_idx_orbital_===0] [idx] ...,   band_spin_atom_orbital_
parse ↑ loop by pre-allocation and popping

curve = [ x1    x2    y ],   curve[0],   curve[x1]

np.float_ for all numeric arrays

allow top-down inheritance.
allow but not recommend data duplication.


Convention
----------

don't over-optimize.

recompute() method + debug_wrap for migration

LBYL to avoid extra indentation.

Use the `custom-fold` package for cross-platform editing.

At the same time, use Atom built-in folding and the fragile `remember-folds` package, for less important tweaks.

Use Atom developer mode to tweak:
- ignore a[1][2] in markdown

Dask parallelization is easy and good. Unfortunately, overheads are of the order of 1 min. Plus, numpy automatically utilizes parallelization.
