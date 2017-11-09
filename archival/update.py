'''
With each update, old objects need to be updated to comply with new API.
This is the latest code depot.
'''

# traverse
for node in engine.Map().lookup('master').map.traverse():
    pass

# update node
# node.default_path = types.MethodType(qchem.Node.__dict__['default_path'], node)
