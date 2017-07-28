from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
from flask import jsonify
from functools import wraps

import sys
import os
from pprint import pprint
import random
import string
from fuzzywuzzy import process
import pickle
from cStringIO import StringIO

import qchem
from shared import *


import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

qchem.Load()    # startup load. wouldn't hurt.

# the modifier
def patch_through(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        sys.stdout = mystdout = StringIO()
        try:
            func(*args, **kwargs)
            sys.stdout  = sys.__stdout__
            return mystdout.getvalue() + '\n' + '-'*30 + '\ngui: +' + func.__name__ + '+ success'
        except CustomError as e:
            sys.stdout  = sys.__stdout__
            return mystdout.getvalue() + '\n' + '-'*30 + '\ngui: ' + func.__name__ + ' failed: ' + str(e) 
    return wrapped

def return_through(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CustomError as e:
            return jsonify( {'error':str(e) } )
    return wrapped


# a random id generator
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

# json translation api
def to_json(n):
    new_json = {}
    # beautify_json
    new_json['label']=new_json['id']=n.name
    palette = {-1: 'A44', 0: '#777',  1: '#863', 2: '#373'}
    new_json['color'] = palette[n.moonphase()]
    # to_json
    for attr_name in vars(n):
        if attr_name != 'map':
            new_json[attr_name] = str( getattr(n,attr_name,None) )
        if attr_name == 'map':
            map_json = {'nodes': [], 'edges':[]}
            for src in n.map._dict:
                map_json['nodes'].append( to_json(src) )
                for dst in n.map._dict[src]:
                    map_json['edges'].append( {'id': id_generator(), 'source': src.name, 'target': dst.name, 'type' : 'arrow', 'color': palette[min(src.moonphase(), dst.moonphase())] } )  # beautify
                for dst in n.map._dict2[src] if src in n.map._dict2 else []:
                    map_json['edges'].append( {'id': id_generator(), 'source': src.name, 'target': dst.name, 'type' : 'dashed', 'color': palette[min(src.moonphase(), dst.moonphase())] } )

            new_json['map'] = map_json
    return new_json


def traverse_json(j, cur_prefix=None):   
    if cur_prefix:
        cur = cur_prefix + '.' + j['name']
    else:
        cur = j['name']
    result = []
    jam = ''
    other = {}
    for key in j:
        if key in ['name','phase','property']:
            jam += ' ' + j[key]
        elif key == 'cell':
            jam += ' ' + j[key].splitlines()[5] + j[key].splitlines()[6]
        elif key in ['id', 'label'] or ':' in key:
            pass
        else:
            other[key] = j[key]
    result.append([cur, jam, other])

    if 'map' in j and 'nodes' in j['map']:
        for subj in j['map']['nodes']:
            result += traverse_json(subj, cur)
    return result
            
def lookup_json(j, cur):
    if cur == 'master':
        return j    # note that the behavior is slightly different from qchem and sigmajs. the reason is that master is not global.
    elif '.' not in cur:
        l = [x for x in j['map']['nodes'] if x['name']==cur]
        if l:
            return l[0]
        else:
            raise KeyError('lookup_json: cannot find %s under current node' %cur)
    else:
        return lookup_json(lookup_json(j,cur.split('.')[0]), '.'.join(cur.split('.')[1:]))

def combine_json(new_json, old_json=None):
    if old_json:
        for newj in traverse_json(new_json):
            if( any([oldj[0]==newj[0] for oldj in traverse_json(old_json)]) ):
                oldj = [oldj for oldj in traverse_json(old_json) if oldj[0]==newj[0]][0]
            else:
                best_jam = process.extractOne(newj[1], [oldj[1] for oldj in traverse_json(old_json)])
                if best_jam[1] > 50:
                    print '*='*50
                    print 'yeah found match'
                    print '-new-'*20
                    print newj
                    print '-old-'*20
                    print oldj
                    print '*='*50
                    oldj = [oldj for oldj in traverse_json(old_json) if oldj[1]==best_jam[0]][0]
                else:
                    oldj = [oldj[0], oldj[1], {}]
                if 'x' in oldj[2] and 'y' in oldj[2]:
                    oldj[2]['x'] = float(oldj[2]['x']) + random.uniform(0,0.1)
                    oldj[2]['y'] = float(oldj[2]['x']) + random.uniform(0,0.1)
            for key in oldj[2]:
                if oldj[2][key] and key not in newj[2]:
                    newj[2][key] = oldj[2][key]
                    lookup_json(new_json,newj[0])[key] = oldj[2][key]
    for newj in traverse_json(new_json):
        tmp = lookup_json(new_json, newj[0])
        if 'x' not in tmp or 'y' not in tmp:
            tmp['x'] = random.uniform(0,1)
            tmp['y'] = random.uniform(0,1)
        if 'read_cam0:x' not in tmp or 'read_cam0:y' not in tmp:
            tmp['read_cam0:x'] = 200 * tmp['x'] - 100
            tmp['read_cam0:y'] = 200 * tmp['y'] - 100
        if 'renderer1:x' not in tmp or 'renderer1:y' not in tmp:
            tmp['renderer1:x'] = 500 * tmp['x'] 
            tmp['read_cam0:y'] = 250 * tmp['y']
    return new_json

@app.route('/reset_', methods=['GET'])
@patch_through
def reset_():
    NODES = {}

@app.route('/import_', methods=['GET'])
@patch_through
def import_():
    with open('data/markdown') as f:
        qchem.Import(f.read())

@app.route('/new_', methods=['GET'])
@patch_through
def new_():
    NODES['master'] = qchem.Node('master')

@app.route('/dump_nodes', methods=['GET'])
@patch_through
def dump_nodes():
    qchem.Dump()

@app.route('/dump_sigma', methods=['POST'])
@patch_through
def dump_sigma():
    old_json = request.get_json(force=True)
    with open(os.path.dirname(os.path.realpath(__file__))+'/data/sigma.dump','wb') as dumpfile:
        pickle.dump(old_json, dumpfile, protocol=pickle.HIGHEST_PROTOCOL)

@app.route('/load_nodes', methods=['GET'])
@patch_through
def load_nodes():
    qchem.Load()

@app.route('/load_sigma', methods=['GET'])
@return_through
def load_sigma():
    with open(os.path.dirname(os.path.realpath(__file__))+'/data/sigma.dump','rb') as dumpfile:
        old_json = pickle.load(dumpfile)
    return jsonify(old_json)

@app.route('/request_', methods=['POST','GET'])
@return_through
def request_():  # either merge json, or use NODES['master']
    if request.method == 'POST':
        old_json = request.get_json(force=True)
        new_json = to_json(NODES['master'])
        new_json = combine_json(new_json, old_json)
        return jsonify( new_json )
    else:
        new_json = to_json(NODES['master'])
        new_json = combine_json(new_json)
        return jsonify(new_json)


@app.route('/add_node', methods=['POST'])
@patch_through
def add_node():
    j = request.get_json(force=True)
    n = NODES['master'].map.lookup(j['cur'])
    n.map.add_node(qchem.Node(j['name']))

@app.route('/del_node', methods=['POST'])
@patch_through
def del_node():
    j = request.get_json(force=True)
    NODES['master'].map.lookup(j['cur']).map.del_node(j['name'])

@app.route('/reset_node', methods=['POST'])
@patch_through
def reset_node():
    j = request.get_json(force=True)
    NODES['master'].map.lookup(j['cur']).reset()

@app.route('/cut_node', methods=['POST'])
@patch_through
def cut_node():
    j = request.get_json(force=True)
    n = NODES['master'].map.lookup(j['cur'])
    NODES[j['name']] = n.map.lookup(j['name'])
    n.map.del_node(j['name'])

@app.route('/paste_node', methods=['POST'])
@patch_through
def paste_node():
    j = request.get_json(force=True)
    n = NODES['master'].map.lookup(j['cur'])
    NODES[j['name']] = n.map.lookup(j['name'])
    n.map.add_node(n.map.lookup(j['name']))

@app.route('/compute_node', methods=['POST'])
@patch_through
def compute_node():
    j = request.get_json(force=True)
    NODES['master'].map.lookup(j['cur']).compute()

@app.route('/edit_vars', methods=['POST'])
@patch_through
def edit_vars():
    j = request.get_json(force=True)
    NODES['master'].map.lookup(j.pop('cur')).edit_vars(j)

@app.route('/get_text', methods=['POST'])
@return_through
def get_text():  
    j = request.get_json(force=True)
    n = NODES['master'].map.lookup(j['cur'])
    return jsonify({'text':str(n)})

@app.route('/edit', methods=['POST'])
@patch_through
def edit():
    j = request.get_json(force=True)
    NODES['master'].map.lookup(j.pop('cur')).edit(j['text'])

@app.route('/check_status', methods=['GET'])
@return_through
def check_status():
    if getattr(shared, 'NODES', None) and 'master' in NODES and getattr(NODES['master'], 'map', None):
        return jsonify({'text':'success'})
    else:
        return jsonify({'text':'warning'})

@app.route('/del_node', methods=['POST'])
@patch_through
def del_node():
    j = request.get_json(force=True)
    n = NODES['master'].map.lookup(j['cur'])
    NODES[j['name']] = n.map.lookup(j['name'])
    n.map.del_node(j['name'])

@app.route('/paste_node', methods=['POST'])
@patch_through
def paste_node():
    j = request.get_json(force=True)
    n = NODES['master'].map.lookup(j['cur'])
    NODES[j['name']] = n.map.lookup(j['name'])
    n.map.add_node(n.map.lookup(j['name']))

@app.route('/compute_node', methods=['POST'])
@patch_through
def compute_node():
    j = request.get_json(force=True)
    NODES['master'].map.lookup(j['cur']).compute()

@app.route('/edit_vars', methods=['POST'])
@patch_through
def edit_vars():
    j = request.get_json(force=True)
    NODES['master'].map.lookup(j.pop('cur')).edit_vars(j)

@app.route('/get_text', methods=['POST'])
@return_through
def get_text():  
    j = request.get_json(force=True)
    n = NODES['master'].map.lookup(j['cur'])
    return jsonify({'text':str(n)})

@app.route('/edit', methods=['POST'])
@patch_through
def edit():
    j = request.get_json(force=True)
    NODES['master'].map.lookup(j.pop('cur')).edit(j['text'])

@app.route('/check_status', methods=['GET'])
@return_through
def check_status():
    if getattr(shared, 'NODES', None) and 'master' in NODES and getattr(NODES['master'], 'map', None):
        return jsonify({'text':'success'})
    else:
        return jsonify({'text':'warning'})


@app.route('/add_edge', methods=['POST'])
@patch_through
def add_edge():
    j = request.get_json(force=True)
    n = NODES['master'].map.lookup(j['cur'])
    n.map.add_edge(j['src'],j['dst'])

@app.route('/del_edge', methods=['POST'])
@patch_through
def del_edge():
    j = request.get_json(force=True)
    n = NODES['master'].map.lookup(j['cur'])
    n.map.del_edge(j['src'],j['dst'])
