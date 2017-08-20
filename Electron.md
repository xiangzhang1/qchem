Electron
========

__log__


# Grepen

__energy__ __efermi__
outcar __nbands__  __nedos__
prev __spin__ __sigma__ __ismear__


# Dos

__log__
__idx_fermi__   dos [idx_fermi] = [ __efermi__ ... ]

__dos__         raw dos.
                dos [idx]  =  _DOSCAR_   =    [ energy    DOS                  IDOS                ]   spin=para
                                              [ energy    DOS(up) DOS(down)    IDOS(up) IDOS(down) ]   spin=fm|afm
                                              [ energy    DOS                  IDOS                ]   spin=ncl
                                        __nspin_dos__   
__dos_interp__() [_idx_spin_] (energy) = DOS

__pdos__        raw pdos.
                pdos [idx_atom + 1] [idx] = [ energy    s-DOS                           p-DOS         ... ]   spin=para
                                            [ energy    s-DOS(up) s-DOS(down)           p-DOS(up)     ... ]   spin=fm|afm
                                            [ energy    s-DOS(x) s-DOS(y) s-DOS(z)      p-DOS(x)      ... ]   spin=fm|afm
                                                        __nspin_pdos__   __norbital_pdos__
__pdos_interp__() [_idx_atom_] [_idx_orbital_] [_idx_spin_] (energy) = DOS

__bandgap__     bandgap[idx_spin] = [] or [vbm, cbm]. compatible with nspin_dos.


# Bands

_EIGENVAL_ =
_# of ions_, _# of ions_ _the number of loops after which the averaged pair correlation functions and the DOS are written_, _ISPIN_
2) _the volume of the cell (in Å^3)_ and the lattice parameters of the box (in m)
3) T
4) the string 'CAR'
5) the header
6) # of electrons, # of k-points, #of bands
7) the k-point and its weight

__nkpt__    total number of reduced kpoints

__fit_neargap_bands__() [_idx_band_] = [ kx    ky    kz    E ]
