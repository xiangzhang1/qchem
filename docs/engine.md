# Part I. Non-nodal engines




Cell
=========================================================================

Parser for POSCAR 5.

__base__ [0] = [ a_x    a_y    a_z ]

__stoichiometry__ = {  _symbol_   :   _natoms_   }

__ccoor__
    __ccoor_kdtree()__
    __ccoor_mindist()__
    __cdist()__
__fcoor()__

__nelectrons()__ # of electrons involved in computation (POTCAR ZVAL)
__natoms()__







Gen
=========================================================================


VASP手册定义了一系列  从 单相->波函数值(->单相/值) + 材料先验参数 + 近似 + 辅助行为 到 参数名列->值列 的映射。

可以用 少量手动执行+gen程序 代替 大量手动执行。

需求分析
------
输入 材料元素 + POSCAR + 求值元素 + 简化近似元素 + 其他近似参数 + 辅助行为参数。如输入正确，输出。如输入无效、冲突、结果不唯一，报错。
允许中间信息。
一般不允许信息丢失。
干净明白。

逻辑设计
------
1. 记元素为mod，记参数为keyword
2. 执行要求

物理实现
------

文件格式：
mod (sections: eval, approx, aux)
require (incl kw/mod, in / !in / complex_funcname, incl rec) [in1 means unique value] [greedy]
functions

流程:
读mod{name:valset=[True]}, kw{name:valset}
	参数值统一用valset限制，string格式，理想情况为单值
执行require
	冲突要求：
		不存在该项 应解释为 尚未有要求
		且/减结果为空 应解释为 冲突  
    条件不满足或冲突，则入栈循环
检验
	输入有效要求：
		modname, kwname被提到过: kw_legal_set, modname_legal_set, kw_internal_set
	唯一结果要求（意义列表）：
		长度为1则合法，不在合法列表内则应无视，其他则为不唯一并报错


Extras
-------
input grammar is require grammar.








Makeparam
=========================================================================
Gathers data to be used in Ml module.

Computes `makeparam` on gen input, except that isym=0 and lsorbit=.false.







# Part II. Node-level engines


Dummy
=========================================================================
Boilerplate.





Compare
=========================================================================
Comparison tools (cell, bands).






Md
=========================================================================
molecular dynamics post-processing toolkit

_data_[idx_traj, idx_atom, idx_step]






Electron
=========================================================================

__log__

总体来说，设计为数据在封装对象之间的传递。


Grepen
------------------------------------------------------------------------

outcar
 __efermi__
__energy__
__nbands__  
__nedos__

prev
__prev_gen__ (avoided)
__spin__
__sigma__
__ismear__

eigenval
__temperature__
__nelectrons__
__nkpts__ # of reduced kpoints in eigenval

__is_doscar_usable__
__is_kpoints_mesh__


Dos
------------------------------------------------------------------------

__log__
__idx_fermi__   dos[][idx_fermi] = [ grepen.efermi ... ]    pdos[][][][idx_fermi] = [ grepen.efermi ... ]

_DOSCAR_ =
len(cell.stoichiometry)             ..                                                  # of loops                  ispin
volume(cell.base)                   norm(cell.base[:])                                  ?
grepen.temperature
'CAR'
cell.name
min_energy              max_energy          nedos           grepen.efermi               1
_DOSCAR_TOT_
min_energy              max_energy          nedos           grepen.efermi               1
_DOSCAR_SITE_TOT_
min_energy              max_energy          nedos           grepen.efermi               1
_DOSCAR_SITE_[_idx_atom_=0]
min_energy              max_energy          nedos           grepen.efermi               1
...

_DOSCAR_TOT_  = [ energy    DOS                  IDOS                ]   spin=para
                [ energy    DOS(up) DOS(down)    IDOS(up) IDOS(down) ]   spin=fm|afm
                [ energy    DOS                  IDOS                ]   spin=ncl
              =  dos[][idx=0][energy]     dos[_idx_spin_==0][idx][DOS]   ...     idos    ...             

__dos__[_idx_spin_][idx] = [ energy DOS ]
__nspins_dos__ spin=para: 1, spin=fm|afm: 2, spin=ncl: 1                                           
__dos_interp__()[_idx_spin_](energy) = DOS
__bandgap__[_idx_spin_] = [vbm, cbm] or []

_DOSCAR_SITE_[_idx_atom_=0] =  [ energy    s-DOS                                      p-DOS         ... ]   spin=para
                               [ energy    s-DOS(up) s-DOS(down)                      p-DOS(up)     ... ]   spin=fm|afm
                               [ energy    s-DOS(x) s-DOS(y) s-DOS(z) s-DOS(tot)      p-DOS(x)      ... ]   spin=fm|afm
                            =  pdos[][_idx_atom_][][idx==0][energy]  pdos[_idx_spin_====0][_idx_atom_][_idx_orbital_===0][idx][PDOS]     ...     ...
                ...
                ...

__pdos__[_idx_spin_][_idx_atom_][_idx_orbital_][idx] = [ energy PDOS ]
__nspins_pdos__ spin=para: 1, spin=fm|afm: 2, spin=ncl: 4   __norbitals_pdos__
__pdos_interp__()[_idx_spin_][_idx_atom_][_idx_orbital_](energy) = DOS


Bands
------------------------------------------------------------------------

_EIGENVAL_ =
len(cell.stoichiometry)             ..                                                  # of loops                  ispin
volume(cell.base)                   norm(cell.base[:])                                  ?
grepen.temperature
'CAR'
cell.name
grepen.nelectrons                   grepen.nkpts                                        grepen.nbands
[empty line]
kpts[_idx_kpt_][0]   kpts[_idx_kpt_][1]   kpts[_idx_kpt_][2]                            weight
_idx_band_==0                       bands[_idx_spin_===0][_idx_band_][_idx_kpt_]  ...   occupancy    ...
... ([empty ... occupancy ...\n)

__kpts__[_idx_kpt_] = [ kx ky kz ]      <!-- less elegant, but saves space -->

__bands__[_idx_spin_][_idx_band_][_idx_kpt_] = E
__nspin_bands__ spin=para: 1, spin=fm|afm: 2, spin=ncl: 1
__bandgaps__[_idx_spin_] = [vbm, cbm] or []     <!-- small. not a numpy array -->

<!-- Interpolate and get bandgap.
Because the region is irregular, the 1st idea is to compute Rbf(*kpt) for each kpt. This costs too much time per kpt.
The 2nd idea is to minimize Rbf, with the constraint that such a 'signed distance' is positive:
  0. Compute Delaunay(points) and ConvexHull(points)
  1. Compute whether point is in Delaunay [](https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl)
  2. Compute distance from point to nearest ConvexHull facet, and multiply by 1's sign.  [](http://mathinsight.org/distance_point_plane)      [](http://scipy.github.io/devdocs/generated/scipy.spatial.ConvexHull.html)
  3. Add by min_kpt_dist
But convergence turned out poor.
Thus, the 3rd and slowest idea is to find maximum and minimum near each kpoint.
-->
<!-- note that any attempt at parallelization requires pickling of entire environment, and is thus unlikely to be efficient -->
__bands_interp__[_idx_spin_][_idx_band_](kx,ky,kz) = E                          |
__bandgaps_interp__                                                             |



Charge
------------------------------------------------------------------------

Errors
------------------------------------------------------------------------

__error__

__min_kpt_dist__      smallest |k1-k2|                                                                              |       __eigenvalue_jump__  
_kpts_nn_           KDTree object storing nearest-neighbor information about __kpts__ array.                        |
_kpts_nn_list_[]    = [idx1_kpt, idx2_kpt] where |k1-k2|~min_kpt_dist                                               |
                                                                                                                    |
<!-- neargap_bands deltaE_deltaKpt -->                                                                              |
_de_dkpt_[idx_spin=0][ZERO==bandgap/2][idx_band===0][kpt_nn_list_====[0,1]] = E <!--doesnt actually exist -->       |
_de_dkpt_flat_[]                                                                                                    |
__de_dkpt__ dE(dkpt)   


gen.getkw('backdrop'), _backdrop_: node with compared property (optimized_cell, bands)
gen.getkw('compare'), _compare_:

    _min_coor_dist_, _coors_nn_, _coors_nn_list_





Vasp
=========================================================================
basics:

__gen__
__cell__ warning: data duplication necessated by modularity
__path__
__prev__

__name__

clerical:

__remote_folder_name__


moonphase, compute:

__wrapper__
__subfile__

if opt:
    __optimized_cell__
