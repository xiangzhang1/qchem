Electron
========

__log__


# Grepen

outcar
 __efermi__
__energy__
__nbands__  
__nedos__

prev
__spin__
__sigma__
__ismear__

eigenval
__temperature__
__nelectrons__
__nkpts__ # of reduced kpoints


# Dos

__log__
__idx_fermi__   dos[][idx_fermi] = [ grepen.efermi ... ]

_DOSCAR_ =
len(cell.stoichiometry)             ..                                                  # of loops                  ispin
volume(cell.base)                   norm(cell.base[:])                                  ?
grepen.temperature
'CAR'
cell.name
min energy              max energy          nedos           grepen.efermi               1
_DOSCAR_TOT_
min energy              max energy          nedos           grepen.efermi               1
_DOSCAR_SITE_TOT_
min energy              max energy          nedos           grepen.efermi               1
_DOSCAR_SITE_[_idx_atom_=0]
min energy              max energy          nedos           grepen.efermi               1
...

_DOSCAR_TOT_  = [ energy    DOS                  IDOS                ]   spin=para
                [ energy    DOS(up) DOS(down)    IDOS(up) IDOS(down) ]   spin=fm|afm
                [ energy    DOS                  IDOS                ]   spin=ncl
              =  dos[][idx=0][energy]     dos[_idx_spin_==0][idx][DOS]   ...     idos    ...             

__dos__[_idx_spin_][idx] = [ energy DOS ]
__nspins_dos__ spin=para: 1, spin=fm|afm: 2, spin=ncl: 1                                           
__dos_interp__()[_idx_spin_](energy) = DOS
__bandgap__[_idx_spin_] = [vbm, cbm] or []

_doscar_site_[_idx_atom_=0] =  [ energy    s-DOS                           p-DOS         ... ]   spin=para
                               [ energy    s-DOS(up) s-DOS(down)           p-DOS(up)     ... ]   spin=fm|afm
                               [ energy    s-DOS(x) s-DOS(y) s-DOS(z)      p-DOS(x)      ... ]   spin=fm|afm
                            =  pdos[][_idx_atom_][][idx==0][energy]  pdos[_idx_spin_====0][_idx_atom_][_idx_orbital_===0][idx][PDOS]     ...     ...
                ...
                ...

__pdos__[_idx_spin_][_idx_atom_][_idx_orbital_][idx] = [ energy PDOS ]
__nspins_pdos__ spin=para: 1, spin=fm|afm: 2, spin=ncl: 3   __norbitals_pdos__
__pdos_interp__()[_idx_spin_][_idx_atom_][_idx_orbital_](energy) = DOS


# Bands

_EIGENVAL_ =
len(cell.stoichiometry)             ..                                                  # of loops                  ispin
volume(cell.base)                   norm(cell.base[:])                                  ?
grepen.temperature
'CAR'
cell.name
grepen.nelectrons                   grepen.nkpts                                        grepen.nbands
[empty line]
bands[][][idx=0][ kx, ky, kz ]      weight
_idx_band_==0                       bands[_idx_spin_===0][_idx_band_][idx][E]  ...   occupancy    ...
...     

__bands__[_idx_spin_][_idx_band_][idx] = [ kx    ky    kz    E ]

__bands_interp__[_idx_spin_][_idx_band_](kx,ky,kz) = E

__idxs_band_neargap__ = []
