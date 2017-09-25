How to handle errors:

RTFM & RTFSC & RTFE. See what it means; try to put together a conclusion

Read the VASP Error Wiki. http://error.wiki/VASP

Google.


===========================================================================
===========================================================================



VERY BAD NEWS! internal error in subroutine IBZKPT:
149 Reciprocal lattice and k-lattice belong to different class of lattices. Often results are still useful...

>

RTFE: That means K-points is not compatible with POSCAR symmetry.
VASP determines symmetry from POSCAR & MAGMOM etc. Then the K-POINTS and charge density is symmetrized (as is requested in standard LDA procedures)

>

the ratios of nk1 nk2 and nk3 have to scale roughly like the inverse of the corresponding legths of the lattice parameters of you cell. Mabye that is the reason.
#why? well, possibly complex machinery issues. use cubic then. and gamm-centered. and nk1 nk2 nk3 proportional

please check if the symmetry of the lattice was preserved during the relaxation runs. the KPOINTS file for the second run must of course be compatible with the symmetry with that lattice (in case the symmetry changed)

>

Error wiki

When this error is legal it means your reciprocal space and sampling grid have a different symmetry. This causes problem in reducing the k-mesh to the irreducible Brillouin zone (and create the IBZKPT file) and thus makes it crash.

So if a=b ensure that kx=ky.

Often however k point grids are compatible (e.g. cubic).

Can often be ignored (but do check whether your k grid looks ok)

Seems to be compilation-related...

possible solution: switch from a Monkhorst pack, to a Gamma centered grid.

>

web:

(triclinic and simple monoclinic as it seems). maybe the axes ratios of the unit cells are not 1:1:1 ? please check.

In any case, if ISYM=0 is set, symmetrization is not taken into account in the calculation later on, the only point to be careful with is the axis' ratio then.

>

The message does not mean an error.
The procedure doing the evaluation of symmetry uses a numerical criterion (SYMPREC=1E-5)
to determine which atoms are symmetrically equivalent. When this criterion is too soft
the programm can find "false" symmetry and reports that "reciprocal lattice and k-lattice belong
to different class of lattices." To get rid of this message you can:
1) Switch off the symmetry (ISYM=0)
2) Make the treshold for detecting the symmetry more severe (SYMPREC=1E-8)

-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------

LAPACK: Routine ZPOTRF failed!

>

RTFM: ZPOTRF factorizes a Hermitian positive definite complex matrix.
Failure may indicate numerical instability. what is it, then?

>

 1) most probably, the forces of the first+second ionic steps lead to an unreasonable geometry for the third step (huge differences in the total energies!), which could not be converged electronically. (ie this is no LAPACK error, but just an unreasonable setup for that ionic step)

        to check:
       --) please have a look at the FORCES on the atoms at the end of ionic step #1 and #2, and the corresponding geometry changes which gave the input geometry for step #3 (the input geometries of each step are listed in XDATCAR). If the ions move too much, the wave function prediction may fail)
       --) please reduce POTIM and/or use a different ionic relaxation algorithm (IBRION).

>

Error wiki:

Occurred during hybrid calculations, try switching between well-converged WAVECARS and CHGCARs. I believe preconverging with PBE to 1e-8 may have fixed it. Custodian halves the potim and deletes chgcar and wavecar. Perhaps lowering the TIME stap may improve stability based on this. I think they may also turn off symmetry but I'd try everything else first if you're doing HSE... I got past it without doing so.

>

Xiang concluding:

Maybe the instability is with the WAVECAR. maybe i should increase WAVECAR prec to, say, 1e-8. And POTIM could be less.

-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------

internal error in CHECK_FULL_KPOINTS: KPOINTS_FULL not properly initialised

>

RTFE: maybe the K-mesh is problematic. Not at the first step; that indicates that the problem is with something changing e.g. charge density

>

RTFM: For ISYM=3, the forces and the stress tensor only are symmetrized, whereas the charge density is left unsymmetrized
(VASP.5.1 only). This option might be useful in special cases, where charge/orbital ordering lowers the crystal symmetry,
and the user wants to conserve the symmetry of the positions during relaxation. However, the flag must be used with great
caution, since a lower symmetry due to charge/orbital ordering, in principle also requires to sample the Brillouin zone using
a k-point mesh compatible with the lower symmetry caused by charge/orbital ordering.

>

Test: change KPOINTS grid to 3 3 2, which is almost the reciprocal of lattice vector lengths, does not work.
Test: change isym to 2, which works

>

Xiang concluding:

So symmetry somehow incompatible? That's acceptable if KPOINTS doesn't have enough symmetry. strange...
 1) use a sensible KPOINTS should help.
 2) avoid isym=3. while it's a speed boost, it inevitably leads to errors even in non-hybrid-functional calculations.

-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------

 -----------------------------------------------------------------------------
|                                                                             |
|           W    W    AA    RRRRR   N    N  II  N    N   GGGG   !!!           |
|           W    W   A  A   R    R  NN   N  II  NN   N  G    G  !!!           |
|           W    W  A    A  R    R  N N  N  II  N N  N  G       !!!           |
|           W WW W  AAAAAA  RRRRR   N  N N  II  N  N N  G  GGG   !            |
|           WW  WW  A    A  R   R   N   NN  II  N   NN  G    G                |
|           W    W  A    A  R    R  N    N  II  N    N   GGGG   !!!           |
|                                                                             |
|      You have selected ISYM=2 for HF type calculation. This will symmetrize |
|      the charge density but not the exchange potential.                     |
|      I suggest to use ISYM=3 instead. This uses symmetry to obtain          |
|      the orbitals at all k-points in the Brillouine zone, but does not      |
|      apply symmetry to the Hartree potential directly.                      |
|      The resultant charge might have lower symmetry than the crystal,       |
|      but at least, Hartree and exchange are fully compatible.               |
|                                                                             |
 -----------------------------------------------------------------------------

 >

 Xiang concluding:

 OK, obviously, if you say so. No ISYM=2 then.


 -------------------------------------------------------------------------------------------------------------------------------------------
 -------------------------------------------------------------------------------------------------------------------------------------------

 EDWAV internal error: the gradient is not orthogonal

 >

 RTFE: wavecar problem.

 >

 Wiki: Seen in hybrid calculations (which hang). Start from a properly preconverged WAVECAR not CHGCAR. It is also POTCAR-dependent.

 The internet suggests wrong intel libraries or overzealous optimization by the compiler may be involved as well.

 >

 First: If the vasp was compiled by intel compiler 2013, then change to 2012 or lower;
 Seceond: change the the optimization flag from -O3 to -O1 .


 -------------------------------------------------------------------------------------------------------------------------------------------
 -------------------------------------------------------------------------------------------------------------------------------------------

 internal error in GENERATE_KPOINTS_TRANS: number of G-vector changed in star

 >

 Usually it is because the NPAR is not set proper, remove "NPAR" from INCAR.

 -------------------------------------------------------------------------------------------------------------------------------------------
 -------------------------------------------------------------------------------------------------------------------------------------------

 internal error in FOCK_ACC: number of k-points incorrect  

 >

 Reason: The number of pre-reduced k-points (in KPOINTS file) should be dividable by NKREDX NKREDY NKREDZ.

 Solution: check the setting of NKRED. (or, if NKRED unset, NKREDX & NKREDY NKREDZ. they are defaulted all to NKRED = 2)

 Note, NKREDX*NKREDY*NKREDZ propto speed.


 -------------------------------------------------------------------------------------------------------------------------------------------
 -------------------------------------------------------------------------------------------------------------------------------------------

 VER BAD NEWS! internal error in subroutine SGRCON: Found some non-integer element in rotation matrix

 >

 Reason unknown. Should be a problem with KPOINTS and symmetry (thus possibly POSCAR)

 Increase SYMPREC might help

 cp IBZKPT to KPOINTS might help

fixing KPOINTS to a higher symmetry may help

-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------

ERROR: while reading WAVECAR, plane wave coefficients changed

>

RTFM: there seems to be an inconsistency in the this-run-fft-mesh and previous-WAVECAR-fft mesh

>

test: for ncl calculation, changing nkred alone does not help.

sometimes an incorrect ISMEAR will cause this error；

please check whether the FFT meshes have changed；

the CHGCAR really is in the working directory at runtime；

the fft meshes of CHGCAR are compatible ；

More verbosely, grep the OUTCAR file from the SCF calculation and find the values of NGXF, NGYF, and NGZF. Set these parameters explicitly in the INCAR file that you use when calculating the band structure (the ICHARG=11 run).

>

as in the non-collinear run all symmetry is reduced, it may well be that some of the PW coefficients have changed. In this case it would be much safer to start with ISTART=0 in order to avoid that coefficients are read from WAVECAR that do not match the basis functions generated for the non-collinear run

>

in noncollinear calculations, you are supposed to only read CHGCAR but not WAVECAR; in LHF calculation, you are only supposed to read WAVECAR(ISTART=1) but not chgcar. HOW FUNNY.

-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------

LSF jobs hung

>

Possible reason 1: do not change any file (externally called or directly used) during execution process; otherwise the job may hung

Possible reason 2: storage quota reached

Possible reason 3: wrong compilation, albeit that a VASP executable that starts properly. VASP 5.2.12 and VASP 5.3.2 does not exhibit similar problem when compiled (like the local one) with ifort11+intel mkl+fftw3.1.2+mpich, but VASP 5.3.5 does.
The golden rule here is to stick to the 'local' makefile.

Possible reason 4: 神仙局

-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------

makeparam behaves like VASP and runs through the entire SCF cycles

>

makeparam is not present at least from vasp.5.2.12. Compare main.F in vasp.4.6.

-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------

USTC Cloud Delete machine/create image/create machine gets stuck

>

Deletion/create image can take 2-4 hours.
'Create machine' can take 24 hours (overnight).

-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------

a loop job containing mpijob only executes once

>

mpijob may always return an ERR value. If a loop continuously read (i.e. do ... done < input.log), the ERR value causes it to exit (disregarding set +e).

Solution: for i in `cat input.log`; do ; done

-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------

the old and the new charge density differ

>

Error wiki:

Setting ISYM = 0 is the only thing I have ever found to be working consistently.

>

RTFM:

Either that the program is wrong, or the iterative solution is wrong, or there are numerical problems (with any part of the program)

>

test:

I found all sorts of strange issues with Re (during 2015summer_first_highthroughput, during 201606_reboot_test). I'd say skip Re whenever possible.

>

web: [http://blog.sciencenet.cn/blog-90975-515644.html]

一直也受争议 (understandable...)

ISYM=0

增大PREC

增大K点

降低编译器优化选项

用ifort10就会出现这个问题，vasp论坛上说这个说的很多了。改用ifort9或者最新的ifort11

We found that there's an error in the way VASP calculates k-points when using the Monkhorst-pack scheme and that the problem would be fixed if we used Gamma centered instead.

-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------

VASP Stuck

>

test: This issue is element-specific. Re, for one, often causes troubles (along with LaBaZnTcO6). Unphysical configuration might also be accountable.

>

guessed from test: Several times a week, file handles in $HOME will change. Thus, if you keep calling some files in $HOME for too long, the program will not be able to continue to read the file, and gets stuck.

Solution: copy file to somewhere not on FS. Maybe mydir=$(mktemp -dt "$0") can help.

Verify: run single vasp job

>

online: maybe it has to do with compilation . If some BLCR package is not installed, MPI may exit on unknown signal 0, but gives no hint.
(ref: https://www.open-mpi.org/community/lists/users/2010/03/12227.php)

Solution: use precompiled package. If using self-compiled package, make sure that it's fine.


-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------

DENTENT: Can't reach specified precision

> http://cms.mpi.univie.ac.at/vasp-forum/viewtopic.php?t=416:

1. Increase KPOINTS.


-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------


WARNING: dimensions on CHGCAR file are different
ERROR: charge density could not be read from file CHGCAR for ICHARG>10

> use same prec
