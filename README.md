# Fragile parts:

qchem.Node.readable_list: which vars to read *input-able*:  ['name','phase','cell','property','map']

sigma.var = config.{{{tmp}}}: which vars to print *physical* (not sigma) : ['name','phase','cell','property','gen','vasp','electron']

gui.traverse_json: [cur, jam, {elsekey:elseval}]: jam recipe *invariable* = [name, phase, property, cell]

gui.*beautify*: name -> id,label. moonphase -> color(bootstrap: gray -> yellow -> green; red)
