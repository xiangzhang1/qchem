import qchem
from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
from flask import jsonify

from cStringIO import StringIO
import sys
from functools import wraps

import random

app = Flask(__name__)
CORS(app)

qchem.Load()

print qchem.NODES

# the modifier
def patch_through(func):
    @wraps(func)
    def patch_through_and_call(*args, **kwargs):
        sys.stdout = mystdout = StringIO()
        try:
            func(*args, **kwargs)
            return mystdout.getvalue() + '\n' + '-'*30 + '\ngui: +' + func.__name__ + '+ success'
        except Exception as e:
            return mystdout.getvalue() + '\n' + str(e) + '\n' + '-'*30 + '\ngui: ' + func.__name__ + ' failed' 
            sys.stdout  = sys.__stdout__
    return patch_through_and_call



# json translation api
def to_json(n):
    new_json = {}
    for attr_name in vars(n):
        if attr_name != 'map':
            new_json[attr_name] = str( getattr(n,attr_name,None) )
        if attr_name == 'map':
            map_json = {'nodes': [], 'edges':[]}
            for n in n.map._dict:
                map_json['nodes'].append( to_json(n) )
                for m in n.map._dict[n] if n in n.map._dict else []:
                    map_json['edges'].append( {'id': id_generator(), 'source': n.name, 'target': m.name, 'type' : 'arrow'} )
                for m in n.map._dict2[n] if n in n.map._dict2 else []:
                    map_json['edges'].append( {'id': id_generator(), 'source': n.name, 'target': m.name, 'type' : 'dotted'} )
            new_json['map'] = map_json
    new_json['label']=new_json['id']=new_json['name']   # beautify
    return new_json

def traverse_json(j, cur_prefix=None):   # [ cur, jam([name, phase, cell, property]), [other property list] ]     
    if cur_prefix:
        cur = cur_prefix + '.' + j['name']
    else:
        cur = j['name']
    result = []
    jam = ''
    other = {}
    for key in j:
        if key in ['name','phase','property']:
            jam += j[key]
        elif key == 'cell':
            jam += j[key].splitlines()[5] + j[key].splitlines()[6]
        elif key == 'map':
            pass
        else:
            other[key] = j[key]
    result.append([cur, jam, other])
    
    for subj in j['nodes']:
        result += traverse_json(subj, cur)

    return result
            
def lookup_json(j, cur):
    if cur == 'master':
        return j    # note that the behavior is slightly different from qchem and sigmajs. the reason is that master is not global.
    elif '.' not in cur:
        return next([x for x in j['map']['nodes'] if x['name']==cur])
    else:
        return lookup(lookup(j,cur.split('.')[0]), '.'.join(cur.split('.')[1:]))

def combine_json(new_json, old_json):
    for newj in traverse_json(new_json):
        if( any([oldj[0]==newj[0] for oldj in traverse_json(old_json)]) ):
            oldj = [oldj for oldj in traverse_json(old_json) if oldj[0]==newj[0]][0]
        else:
            best_jam = process.extractOne(newj[2], [oldj[2] for oldj in traverse_json(old_json)])
            if best_jam[1] > 50:
                oldj = [oldj for oldj in traverse_json(old_json) if oldj[2]==best_jam[0]][0]
            else:
                oldj = [oldj[0], oldj[1], {'x': random.uniform(0,1), 'y': random.uniform(0,1)}]
        for key in oldj[2]:
            if oldj[2][key] and key not in newj[2]:
                newj[2][key] = oldj[2][key]
                lookup(new_json,newj[0])[key] = oldj[2][key]



@app.route('/load', methods=['POST', 'GET'])
@patch_through
def load():
    qchem.Load()

@app.route('/dump', methods=['POST', 'GET'])
@patch_through
def dump():
    qchem.Dump()

@app.route('/test_import', methods=['POST', 'GET'])
@patch_through
def test_import():
    with open('data/markdown') as f:
        qchem.Import(f.read())

@app.route('/refresh', methods=['POST'])
def refresh():
    old_json = request.get_json(force=True)
    new_json = to_json(qchem.NODES['master'])
    new_json = combine_json(new_json, old_json)
    return jsonify( new_json )
