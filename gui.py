#!/usr/bin/python
'''GUI and CLI, combined in one'''
# flask
from flask import Flask, request, jsonify, session, Response
from flask_cors import CORS, cross_origin
from functools import wraps

# common libraries
import sys
sys.dont_write_bytecode = True
import os
import shutil
import random
import string
import dill as pickle
import time
from pprint import pprint
import IPython
import re
import threading

from cStringIO import StringIO
from fuzzywuzzy import process

# qchem package
import qchem
import shared
import engine
from shared import ELEMENTS


# logging
import logging
if shared.DEBUG <= 1:
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
# logging.basicConfig(filename='error.log',level=logging.DEBUG)
# class NoParsingFilter(logging.Filter):
#    def filter(self, record):
#        return not '/make_connection' in record.getMessage()
# log.addFilter(NoParsingFilter())

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

# patches output and expected CustomError through; login security
def patch_through(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if shared.DEBUG >= 0:
            func(*args, **kwargs)
            return jsonify( {'status':'debug'} )
        else:
            sep = '\n' + '-' * 30 + '\n'
            septop = '=' * 30 + '\n'
            sepbot = '\n' + '=' * 30
            sys.stdout = mystdout = StringIO()
            try:
                func(*args, **kwargs)
                sys.stdout  = sys.__stdout__
                return '%s %s %s gui.py: %s success %s' %( septop, mystdout.getvalue(),  sep,  func.__name__ , sepbot)
            except shared.CustomError as e:  # or shared.CustomError
                sys.stdout  = sys.__stdout__
                return '%s %s %s %s %s gui.py: %s failed %s' %(  septop,  mystdout.getvalue(),  sep,  e,  sep,   func.__name__ , sepbot )
    return wrapped

def return_through(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if shared.DEBUG >= 0 :
            return func(*args, **kwargs)
        else:
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
        if not auth or auth.username!='xzhang1' or auth.password!='5vRPz7Ngm8rNS3Sg':
            return Response('Bad username/pswd', 403, {'WWW-Authenticate': 'Basic realm="Need login."'})
        else:
            return func(*args, **kwargs)
    return wrapped

# for testing
@app.route('/hello_world')
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
                    map_json['edges'].append( {'id': id_generator(), 'source': src.name, 'target': dst.name, 'type' : 'tapered', 'color': shared.COLOR_PALETTE[min(src.moonphase(), dst.moonphase())] } )
            new_json['map'] = map_json
    return new_json

def traverse_json(j, cur_prefix=None):      # returns a triplet: [cur,   jam_for_fuzzy_match,   [graphical properties to transfer]    ]
    if cur_prefix:
        cur = cur_prefix + '.' + j['name']
    else:
        cur = j['name']
    result = []
    jam = ''
    other = {}
    for key in j:
        # cur
        # jam
        if key in ['name','phase','property']:
            jam += ' ' + j[key]
        elif key == 'cell':
            jam += ' ' + j[key].splitlines()[5] + j[key].splitlines()[6]
        # graphical properties
        #wtf#elif key in ['id', 'label'] or ':' in key:
        #wtf#   pass
        elif key in shared.ALL_ATTR_LIST:
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
                    #print '*='*50
                    #print 'yeah found match'
                    #print '-new-'*20
                    #print newj
                    #print '-old-'*20
                    #print oldj
                    #print '*='*50
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
    for fname in os.listdir(shared.SCRIPT_DIR+'/docs'):
        j['filenames'].append('docs/'+fname)
    for fname in os.listdir(shared.SCRIPT_DIR+'/conf'):
        j['filenames'].append('conf/'+fname)
    return jsonify(j)

@app.route('/open_docs', methods=['POST'])
@return_through
@login_required
def open_docs():
    fname = request.get_json(force=True)['fname']
    filename = shared.SCRIPT_DIR+'/'+fname
    with open(filename,'r') as f:
        return jsonify({'text':f.read()})

@app.route('/ipython', methods=['GET'])
@patch_through
@login_required
def ipython():
    '''invokes embedded ipython'''
    IPython.embed()


# the real qchem functions
@app.route('/reset_NODES', methods=['GET'])
@patch_through
@login_required
def reset_NODES():
    shared.NODES = {}

@app.route('/import_markdown', methods=['GET'])
@patch_through
@login_required
def import_markdown():
    with open('data/markdown') as f:
        qchem.Import(f.read())

@app.route('/new_', methods=['GET'])
@patch_through
@login_required
def new_():
    shared.NODES['master'] = qchem.Node('# master\n\nmap:\n\n')

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
    with open(shared.SCRIPT_DIR+'/data/sigma.dump.'+time.strftime('%Y%m%d%H%M%S'),'wb') as dumpfile:
        pickle.dump(old_json, dumpfile, protocol=pickle.HIGHEST_PROTOCOL)


# either load latest, or load a specific datetime.
@app.route('/load_nodes', methods=['GET','POST'])
@patch_through
@login_required
def load_nodes():
    if request.method == 'POST':
        datetime = request.get_json(force=True)['datetime']
        qchem.Load(datetime)
    else:
        qchem.Load()

@app.route('/load_sigma', methods=['GET','POST'])
@return_through
@login_required
def load_sigma():
    if request.method == 'POST':  # used in conjunction with load_nodes, so expect small timestamp difference
        datetime = int(request.get_json(force=True)['datetime'])
        l = [int(x.replace('sigma.dump.','')) for x in os.listdir(shared.SCRIPT_DIR + '/data/') if x.startswith('sigma.dump')]
        if not l:   raise shared.CustomError('Load: no file near {%s} found' %datetime)
        l.sort()
        datetime = str( [x for x in l if abs(x-datetime)<2.1][-1] )
        filename = shared.SCRIPT_DIR + '/data/sigma.dump.' + datetime
    else:
        l = [x for x in os.listdir(shared.SCRIPT_DIR + '/data/') if x.startswith('sigma.dump')]
        if not l:   raise shared.CustomError('Load: no file to load')
        l.sort()
        filename = shared.SCRIPT_DIR + '/data/' + l[-1]
    if os.path.isfile(filename):
        with open(filename,'rb') as dumpfile:
            old_json = pickle.load(dumpfile)
        print 'Loaded {%s}' %filename
    else:
        raise shared.CustomError( 'load_sigma: File {%s} not found' %filename )
        old_json = {}
    return jsonify(old_json)

@app.route('/get_dumps_list', methods=['GET'])
@return_through
@login_required
def get_dumps_list():
    j = {'datetimes':[]}
    l = []
    for fname in os.listdir(shared.SCRIPT_DIR+'/data/'):
        if fname.startswith('shared.NODES.dump.'):
            l.append(fname.replace('shared.NODES.dump.',''))
    l.sort(reverse=True)
    j['datetimes'] = l[:5]
    return jsonify(j)



@app.route('/request_', methods=['POST','GET'])
@return_through
@login_required
def request_():  # either merge json, or use shared.NODES['master']     # yep, this is the magic function.
    if request.method == 'POST':
        old_json = request.get_json(force=True)
        if shared.DEBUG >= 2: print 'before to_json' + '*'*70
        new_json = to_json(engine.Map().lookup('master'))
        if shared.DEBUG >= 2: print 'after to_json' + '*'*70
        new_json = combine_json(new_json, old_json)
        return jsonify( new_json )


@app.route('/new_node', methods=['POST'])
@patch_through
@login_required
def new_node():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    n.map.add_node(qchem.Node())

@app.route('/del_node', methods=['POST'])
@patch_through
@login_required
def del_node():
    j = request.get_json(force=True)
    node = engine.Map().lookup(j['cur']+'.'+j['name'])
    node.delete()

@app.route('/reset_node', methods=['POST'])
@patch_through
@login_required
def reset_node():
    j = request.get_json(force=True)
    engine.Map().lookup(j['cur']).reset()

@app.route('/copy_ref', methods=['POST'])
@patch_through
@login_required
def copy_ref():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    shared.NODES[n.name] = n

@app.route('/duplicate_node', methods=['POST'])
@patch_through
@login_required
def duplicate_node():
    j = request.get_json(force=True)
    parent_n = engine.Map().lookup(j['cur'])
    n = engine.Map().lookup(j['cur']+'.'+j['name'])
    new_node = qchem.Node('# newnode')
    for attr in vars(n):
        if attr == 'name':  # test1 -> test2
            number = int(re.search(r'\d+$', n.name).group(0)) if re.search(r'\d+$', n.name) else 0
            text = re.sub(r'\d+$', '', n.name)
            while True:
                try:
                    engine.Map().lookup(j['cur'] + '.' + text + str(number))
                except LookupError:
                    break
                else:
                    number += 1
            new_node.name = text + str(number)
        elif attr == 'path':
            pass
        elif attr in shared.READABLE_ATTR_LIST:
            setattr(new_node, attr, getattr(n,attr))
    parent_n.map.add_node(new_node)

@app.route('/compute_node', methods=['POST'])
@patch_through
@login_required
def compute_node():
    j = request.get_json(force=True)
    engine.Map().lookup(j['cur']).compute(proposed_name=j['name'])  #delegate to parent, suggest compute name

@app.route('/setinterval_compute_node', methods=['POST'])
@patch_through
@login_required
def setinterval_compute_node():
    def setinterval_compute_node_base(j=request.get_json(force=True)):
        print j['cur'], '8' * 100
        shared.timer = threading.Timer(6, setinterval_compute_node)
        shared.timer.start()
    setinterval_compute_node_base()

@app.route('/stop_setinterval_compute_node', methods=['GET'])
@patch_through
@login_required
def setinterval_compute_node():
    shared.timer.stop()

@app.route('/get_text', methods=['POST'])
@return_through
@login_required
def get_text():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    return jsonify({'text':str(n)})


@app.route('/edit_vars', methods=['POST'])
@patch_through
@login_required
def edit_vars():
    # Node() format input. Less powerful than edit, but more intuitive.
    # Updates mentioend and readable parts only.
    j = request.get_json(force=True)
    n = engine.Map().lookup(j.pop('cur'))
    for name,value in j.iteritems():
        if name == 'name' and ('.' in value or ',' in value or '=' in value):
            raise shared.CustomError('dot, comma and equal sign cannot be in name. ')
        if name not in shared.READABLE_ATTR_LIST:
            continue
        if getattr(engine, name.title(), None):
            value = getattr(engine, name.title())(value)
        setattr(n, name, value)

@app.route('/del_attr', methods=['POST'])
@patch_through
@login_required
def del_attr():
    # get cur and attr_name, and delete
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    delattr(n, j['attr_name'])

@app.route('/edit', methods=['POST'])
@patch_through
@login_required
def edit():
    # Import format input. Not in place.
    # Update mentioned parts only
    j = request.get_json(force=True)
    node = engine.Map().lookup(j.pop('cur'))
    #  map rule
    if 'map:' in j['text']:
        for x in node.map if getattr(node,'map',None) else []:
            if x.name == node.name:
                raise shared.CustomError('gui edit: edit is not in place and relies on name search. one child node has same name {%s} as parent, which may cause confusion.' %node.name)
            shared.NODES[x.name] = x
    qchem.Import(j['text'])
    #  update
    if node.name in shared.NODES:
        new_node = node.map.lookup(node.name)#pop
    else:
        raise shared.CustomError(node.__class__.__name__ + ': edit: You have not defined a same-name node (aka node with name %s which would have been read)' %(node.name))
    for varname in vars(new_node):
        setattr(node, varname, getattr(new_node, varname))


@app.route('/make_connection', methods=['GET'])
@return_through
@login_required
def make_connection():
    if 'master' in shared.NODES and getattr(shared.NODES['master'], 'map', None):
        statuscolor = shared.COLOR_PALETTE[2]
    else:
        statuscolor = shared.COLOR_PALETTE[-1]
    return jsonify({'statuscolor':statuscolor, 'ALL_ATTR_LIST': shared.ALL_ATTR_LIST, 'READABLE_ATTR_LIST': shared.READABLE_ATTR_LIST, 'DEBUG': shared.DEBUG})

@app.route('/paste_ref', methods=['POST'])
@patch_through
@login_required
def paste_ref():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    shared.NODES[j['name']] = n.map.lookup(j['name'])
    n.map.add_node(n.map.lookup(j['name']))

@app.route('/add_edge', methods=['POST'])
@patch_through
@login_required
def add_edge():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    n.map.add_edge(j['src'],j['dst'])

@app.route('/del_edge', methods=['POST'])
@patch_through
@login_required
def del_edge():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    n.map.del_edge(j['src'],j['dst'])

@app.route('/copy_remote_folder_name', methods=['POST'])
@return_through
@login_required
def copy_remote_folder_name():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    try:
        return jsonify({'remote_folder_name': n.vasp.remote_folder_name})
    except:
        return jsonify({'remote_folder_name':'remote_folder_name_not_assigned'})

@app.route('/copy_path', methods=['POST'])
@return_through
@login_required
def copy_path():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    try:
        return jsonify({'path': n.path})
    except:
        return jsonify({'path':'path_not_assigned'})


# ===========================================================================

#app.run(host='127.0.0.1',port='5000', debug = False, ssl_context=('cert/domain.crt','cert/domain.key'))
app.run(host='0.0.0.0', port='5000', debug=False)
