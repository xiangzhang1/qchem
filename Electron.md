Electron
========

__log__


# Grepen

__energy__ __efermi__
outcar __nbands__  __nedos__
prev __spin__ __sigma__ __ismear__


# Dos

__log__

__nspin__       max of _idx_spin_ in dos
__idx_fermi__   dos[idx_fermi][0] = __efermi__

__dos__         raw dos.
                dos[] = [e __dos_interp()__[_idx_spin_](_e_) __idos_interp()__[_idx_spin_](_e_) ...]
__pdos__        raw pdos.

__bandgap__     bandgap[_idx_spin_] = [] or [vbm, cbm]


- __idx_fermi__ Fermi level index for dos and pdos
- __dos__. para, ncl: e dos[0] integrated_dos[0]
                 fm, afm: e dos[0] integrated_dos[0] dos[1] integrated_dos[1]
- __pdos__.
