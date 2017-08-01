from flask import Flask, request, jsonify, session
from flask_cors import CORS, cross_origin
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
import shared
from shared import ELEMENTS

import logging
#log = logging.getLogger('werkzeug')
#log.setLevel(logging.ERROR)
logging.basicConfig(filename='error.log',level=logging.DEBUG)

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

# patches output and expected CustomError through; login security
def patch_through(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
            sys.stdout = mystdout = StringIO()
#        try:
            func(*args, **kwargs)
            sys.stdout  = sys.__stdout__
            return mystdout.getvalue() + '\n' + '-'*30 + '\ngui.py: +' + func.__name__ + '+ success'
#        except shared.CustomError as e:  # or shared.CustomError
#            sys.stdout  = sys.__stdout__
#            return mystdout.getvalue() + '\n' + '-'*30 + '\ngui.py: ' + func.__name__ + ' exception: ' + str(e) 
    return wrapped

def return_through(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except shared.CustomError as e:
            return jsonify( {'error':str(e) } )
    return wrapped

# login security
def login_required(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        # auth
        auth = request.authorization
        if not auth or auth.username!='xzhang1' or auth.password!='yror':
            return Response('Bad username/pswd', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
        else:
            return func(*args, **kwargs)
    return wrapped

# for testing
@app.route('/helloworld', methods=['GET'])
def hellowworld():
    return 'hello, world!'

# a random id generator
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

# json translation api
def to_json(n):
    new_json = {}
    # beautify_json
    new_json['label']=new_json['id']=n.name
    new_json['color'] = shared.COLOR_PALETTE[n.moonphase()]
    # to_json
    for attr_name in vars(n):
        if attr_name != 'map':
            new_json[attr_name] = str( getattr(n,attr_name,None) )
        if attr_name == 'map':
            map_json = {'nodes': [], 'edges':[]}
            for src in n.map._dict:
                map_json['nodes'].append( to_json(src) )
                for dst in n.map._dict[src]:
                    map_json['edges'].append( {'id': id_generator(), 'source': src.name, 'target': dst.name, 'type' : 'arrow', 'color': shared.COLOR_PALETTE[min(src.moonphase(), dst.moonphase())] } )  # beautify
                for dst in n.map._dict2[src] if src in n.map._dict2 else []:
                    map_json['edges'].append( {'id': id_generator(), 'source': src.name, 'target': dst.name, 'type' : 'dashed', 'color': shared.COLOR_PALETTE[min(src.moonphase(), dst.moonphase())] } )

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

# clericals: docs, execute python function
@app.route('/get_docs_list', methods=['GET'])
@return_through
@login_required
def get_docs_list():
    j = {'filenames':[]}
    for fname in os.listdir(os.path.dirname(os.path.realpath(__file__))+'/docs'):
        if fname.endswith('.md'):
            j['filenames'].append(fname)
    return jsonify(j)

@app.route('/open_docs', methods=['POST'])
@return_through
@login_required
def open_docs():
    fname = request.get_json(force=True)['fname']
    filename = os.path.dirname(os.path.realpath(__file__))+'/docs/'+fname
    with open(filename,'r') as f:
        return jsonify({'text':f.read()})

@app.route('/x93d_python_rrho', methods=['POST'])
@patch_through
@login_required
# Tries to eval, then exec.
# If code contains SyntaxError, then it's executed.
def x93d_python_rrho():
    code = request.get_json(force=True)['code']
    try:
        print eval(code)
    except SyntaxError:
        exec(code)

# the real qchem functions
@app.route('/reset_', methods=['GET'])
@patch_through
@login_required
def reset_():
    shared.NODES = {}

@app.route('/import_', methods=['GET'])
@patch_through
@login_required
def import_():
    with open('data/markdown') as f:
        qchem.Import(f.read())

@app.route('/new_', methods=['GET'])
@patch_through
@login_required
def new_():
    shared.NODES['master'] = qchem.Node('master\n\nmap:\n\n')

@app.route('/dump_nodes', methods=['GET'])
@patch_through
@login_required
def dump_nodes():
    qchem.Dump()

@app.route('/dump_sigma', methods=['POST'])
@patch_through
@login_required
def dump_sigma():
    old_json = request.get_json(force=True)
    with open(os.path.dirname(os.path.realpath(__file__))+'/data/sigma.dump','wb') as dumpfile:
        pickle.dump(old_json, dumpfile, protocol=pickle.HIGHEST_PROTOCOL)

@app.route('/load_nodes', methods=['GET'])
@patch_through
@login_required
def load_nodes():
    qchem.Load()

@app.route('/load_sigma', methods=['GET'])
@return_through
@login_required
def load_sigma():
    filename = os.path.dirname(os.path.realpath(__file__))+'/data/sigma.dump'
    if os.path.isfile(filename):
        with open(filename,'rb') as dumpfile:
            old_json = pickle.load(dumpfile)
        print 'Loaded sigma.dump'
    else:
        print 'No sigma.dump to load'
        old_json = {}
    return jsonify(old_json)

@app.route('/request_', methods=['POST','GET'])
@return_through
@login_required
def request_():  # either merge json, or use shared.NODES['master']
    if request.method == 'POST':
        old_json = request.get_json(force=True)
        new_json = to_json(shared.NODES['master'])
        new_json = combine_json(new_json, old_json)
        return jsonify( new_json )
    else:
        new_json = to_json(shared.NODES['master'])
        new_json = combine_json(new_json)
        return jsonify(new_json)


@app.route('/add_node', methods=['POST'])
@patch_through
@login_required
def add_node():
    j = request.get_json(force=True)
    n = shared.NODES['master'].map.lookup(j['cur'])
    n.add_node(qchem.Node(j['name']))

@app.route('/del_node', methods=['POST'])
@patch_through
@login_required
def del_node():
    j = request.get_json(force=True)
    shared.NODES['master'].map.lookup(j['cur']).map.del_node(j['name'])

@app.route('/reset_node', methods=['POST'])
@patch_through
@login_required
def reset_node():
    j = request.get_json(force=True)
    shared.NODES['master'].map.lookup(j['cur']).reset()

@app.route('/cut_node', methods=['POST'])
@patch_through
@login_required
def cut_node():
    j = request.get_json(force=True)
    n = shared.NODES['master'].map.lookup(j['cur'])
    shared.NODES[j['name']] = n.map.lookup(j['name'])
    n.map.del_node(j['name'])

@app.route('/copy_node', methods=['POST'])
@patch_through
@login_required
def copy_node():
    j = request.get_json(force=True)
    n = shared.NODES['master'].map.lookup(j['cur'])
    shared.NODES[j['name']] = n.map.lookup(j['name'])

@app.route('/compute_node', methods=['POST'])
@patch_through
@login_required
def compute_node():
    j = request.get_json(force=True)
    shared.NODES['master'].map.lookup(j['cur']).compute()

@app.route('/edit_vars', methods=['POST'])
@patch_through
@login_required
def edit_vars():
    j = request.get_json(force=True)
    shared.NODES['master'].map.lookup(j.pop('cur')).edit_vars(j)

@app.route('/edit_path', methods=['POST'])
@patch_through
@login_required
def edit_path():
    j = request.get_json(force=True)
    setattr(shared.NODES['master'].map.lookup(j['cur']), 'path', j['path'])

@app.route('/get_text', methods=['POST'])
@return_through
@login_required
def get_text():  
    j = request.get_json(force=True)
    n = shared.NODES['master'].map.lookup(j['cur'])
    return jsonify({'text':str(n)})

@app.route('/edit', methods=['POST'])
@patch_through
@login_required
def edit():
    j = request.get_json(force=True)
    shared.NODES['master'].map.lookup(j.pop('cur')).edit(j['text'])

@app.route('/check_status', methods=['GET'])
@return_through
@login_required
def check_status():
    if 'master' in shared.NODES and getattr(shared.NODES['master'], 'map', None):
        return jsonify({'color':shared.COLOR_PALETTE[2]})
    else:
        return jsonify({'color':shared.COLOR_PALETTE[-1]})

@app.route('/paste_node', methods=['POST'])
@patch_through
@login_required
def paste_node():
    j = request.get_json(force=True)
    n = shared.NODES['master'].map.lookup(j['cur'])
    shared.NODES[j['name']] = n.map.lookup(j['name'])
    n.map.add_node(n.map.lookup(j['name']))

@app.route('/add_edge', methods=['POST'])
@patch_through
@login_required
def add_edge():
    j = request.get_json(force=True)
    n = shared.NODES['master'].map.lookup(j['cur'])
    n.map.add_edge(j['src'],j['dst'])

@app.route('/del_edge', methods=['POST'])
@patch_through
@login_required
def del_edge():
    j = request.get_json(force=True)
    n = shared.NODES['master'].map.lookup(j['cur'])
    n.map.del_edge(j['src'],j['dst'])
