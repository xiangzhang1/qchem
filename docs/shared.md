Ket
===================================================================================


script [axm, meta-axm, state-machine] 
------------------------
our goal is: simplify, separate logic / cleric
map. input: phase, cell, property_wanted {logic [cleric(approximations, engine)]}. output: {property} % wf
map. input: logic (phase, cell, property), cleric (approximations). output: [cleric entity] % gen
map. def. input: -. output: {property}. % class

meta <--> parent: phase, cell, property wanted ==[ child: phase, cell, property wanted --> engine (draw data from)
parent - property_wanted - child - engine - filesystem

axiomatic input. rigorous program. 
key is tree label. save point. one way inheritance.

init -> compute -> inquire on property

ket computes: 
	ket.compute() 
		property_wanted.compute() for phantom
			property_wanted.compute() 
				ket.moonphase() //3:api-finished, 2:api-hang, 1: api-computable, 0: no-api, -1: error
			create Ket(), ket.compute()
		compute_engine.compute() for copy
			self.gen
			self.compute_engine
			self.compute_engine.compute() 
	self.edit(), ket.delete()
		property_wanted.write_(), property_wanted.read_()
ket has properties we can inquire:
		ket.consider()
		
phase.py
	Element, ElementsDict, ELEMENTS, lazyattr, word_wrap
	Ket, KETS
		phase, cell, property_wanted
			* irrelevantly: path, label
		qd, gen, vasp
		compute()
engines.py
	Cell
		name, stoichiometry, nion, nelect, base, coordinates
	Property_wanted: a graphic graph
		nodes[0]: [property_wanted_, , ]. edges[0]: [p_, p_, arrow]
			* irrelevantly: node[1]=path, node[2]=label, edge[0,1] in [property_wanted_, path, label]
			* input_: csv(node) \n\n print(edge)
		compute(), moonphase(), write_()read_(), node(any), prev(any), subset([any])
	Gen: text parser
		mod, kw
		parse_if(), parse_require(), getkw()
		__init__('mod, kw=val',cell)
			engine.gen.conf
	Vasp
		__init__(gen, cell, path)
		moonphase(), compute()
		
	
	
	
	
	
	











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
    
