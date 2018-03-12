Global constants, methods, and (persisting) variables. Note that np.pi is not capitalized.

__DEBUG__
* -1 [silent]
* 0 [auto]
* 1 [manual]
* 2 [verbose interactive debug]

@MWT: timeout = 10 (ssh) | 2592000 (cell.cdist())

__CustomError__, __DeferError__: data is inconsistent. sometimes it simply means 'recalculate, man'.


Element, ELEMENTS
=========================================================================

Purpose
-------
Properties of the chemical elements.

Each chemical element is represented as an object instance. Physicochemical
and descriptive properties of the elements are stored as instance attributes.

Examples
--------
>>> from elements import ELEMENTS
>>> len(ELEMENTS)
109
>>> str(ELEMENTS[109])
'Meitnerium'
>>> ele = ELEMENTS['C']
>>> ele.number, ele.symbol, ele.name, ele.eleconfig
(6, 'C', 'Carbon', '[He] 2s2 2p2')
>>> ele.eleconfig_dict
{(1, 's'): 2, (2, 'p'): 2, (2, 's'): 2}
>>> sum(ele.mass for ele in ELEMENTS)
14659.1115599
>>> for ele in ELEMENTS:
...     ele.validate()
...     ele = eval(repr(ele))


class ElementsDict(object):
---------------------------
Ordered dict of Elements with lookup by number, symbol, and name.


class Element(object):
----------------------
number : int
    Atomic number
symbol : str of length 1 or 2
    Chemical symbol
name : str
    Name in english
group : int
    Group in periodic table
period : int
    Period in periodic table
block : int
    Block in periodic table
series : int
    Index to chemical series
protons : int
    Number of protons
neutrons : int
    Number of neutrons in the most abundant naturally occurring stable
    isotope
nominalmass : int
    Mass number of the most abundant naturally occurring stable isotope
electrons : int
    Number of electrons
mass : float
    Relative atomic mass. Ratio of the average mass of atoms
    of the element to 1/12 of the mass of an atom of 12C
exactmass : float
    Relative atomic mass calculated from the isotopic composition
eleneg : float
    Electronegativity (Pauling scale)
covrad : float
    Covalent radius in Angstrom
atmrad :
    Atomic radius in Angstrom
vdwrad : float
    Van der Waals radius in Angstrom
tboil : float
    Boiling temperature in K
tmelt : float
    Melting temperature in K
density : float
    Density at 295K in g/cm3 respectively g/L
oxistates : str
    Oxidation states
eleaffin : float
    Electron affinity in eV
eleconfig : str
    Ground state electron configuration
eleconfig_dict : dict
    Ground state electron configuration (shell, subshell): electrons
eleshells : int
    Number of electrons per shell
ionenergy : tuple
    Ionization energies in eV
isotopes : dict
    isotope mass number: (relative atomic mass, abundance, isotope mass number)
