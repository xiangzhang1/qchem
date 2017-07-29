our goal is: simplify; separate logic / cleric
map. input: phase, cell, property_wanted {logic cleric}. output: {property}

meta <--> parent: phase, cell, property wanted ==[ child: phase, cell, property wanted --> engine (draw data from)

path and label are irrelevant. separate them to second stage. 合式.
irrelevant (>) vs relevant+ (#)

namespace is path.

save point mechanism.

parent - property_wanted - child - engine - filesystem





Ket:
phase, cell, property_wanted
	path, label
compute()
	gen -> kw['engine']
	
Property_wanted: a graphic graph. input_: property_wanted_ br property_wanted_ -l-> property_wanted_
		property_wanted_ : path : label
		property_wanted_ -> any
	property_wanted_
		link to Ket (path)
		label
	nodes [ [property_wanted_,,] ], edges [ [ property_wanted_, property_wanted_, arrow_ ] ]. 
		nodes [ [property_wanted_, path, label] ]
	compute(), moonphase(), write_()read_(), node(input_), prev(input_), subset(list_)