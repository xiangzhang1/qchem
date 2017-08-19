Electron
========

__log__


# Grepen

__energy__ __efermi__
outcar __nbands__  __nedos__
prev __spin__ __sigma__ __ismear__


# Dos

__log__

__idx_fermi__  dos[idx_fermi][0] = efermi
__dos__        raw dos.
               para or ncl `dos[idx_fermi] = [e dos[0](e) idos[0](e)]`
               fm or afm:  `dos[idx_fermi] = [e dos[0](e) idos[0](e) dos[1](e) idos[1](e)]`
__pdos__       raw pdos.


- __idx_fermi__ Fermi level index for dos and pdos
- __dos__. para, ncl: e dos[0] integrated_dos[0]
                 fm, afm: e dos[0] integrated_dos[0] dos[1] integrated_dos[1]
- __pdos__.
