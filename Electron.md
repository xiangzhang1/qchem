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
__idx_fermi__   dos [idx_fermi] = [ __efermi__ ... ]

_DOSCAR_  =  [ energy    DOS                  IDOS                ]   spin=para
             [ energy    DOS(up) DOS(down)    IDOS(up) IDOS(down) ]   spin=fm|afm
             [ energy    DOS                  IDOS                ]   spin=ncl

__dos__ [_idx_spin_] [idx] = [ energy DOS ]
__nspins_dos__                                            
__dos_interp__() [_idx_spin_] (energy) = DOS
__bandgap__ [_idx_spin_] = [vbm, cbm] or []

_DOSCAR_ATOM_ =  [ energy    s-DOS                           p-DOS         ... ]   spin=para
                 [ energy    s-DOS(up) s-DOS(down)           p-DOS(up)     ... ]   spin=fm|afm
                 [ energy    s-DOS(x) s-DOS(y) s-DOS(z)      p-DOS(x)      ... ]   spin=fm|afm

__pdos__ [_idx_spin_] [_idx_atom_] [_idx_orbital_] [idx] = PDOS
__nspins_pdos__   __norbitals_pdos__
__pdos_interp__() [_idx_spin_] [_idx_atom_] [_idx_orbital_] (energy) = DOS


# Bands

_EIGENVAL_ =
len(cell.stoichiometry)                   ..                                # of loops                  ispin
volume(cell.base)                         norm(cell.base[:])          ?
grepen.temperature
'CAR'
cell.name
grepen.nelectrons                         grepen.nkpts                      grepen.nbands
[empty line]
kpts [_idx_kpt_] [:]
bands [_idx_band_] [_idx_kpt_] [:]

__kpts__ [_idx_kpt_] = [ kx   ky   kz  weight ]   
__bands__ [_idx_band_] [_idx_kpt_] [_idx_spin_] = [  ]
__fit_neargap_bands__() [_idx_band_] = [ kx    ky    kz    E ]
