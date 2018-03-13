#!/usr/bin/env python

# flask
from flask import Flask, request, jsonify, session, Response
from flask_cors import CORS, cross_origin
from functools import wraps
from flask_compress import Compress

# common libraries
import sys
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

from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

import atexit
import thread
import hashlib

# qchem package
from qchem import shared, dynamic, engine, graph

# logging
import logging
logging.basicConfig()


'''
gui.py
=========================================
- Flask basic config, cron-like apscheduler
- @login_required: Login security decorator
- gui-sigma API: all sorts of shitty translations
- A lot of wrappers
- Run flask
'''


# Flask basic config
# ======================================================================

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
#Compress(app)   # compress respons#e

# cron server using apscheduler
scheduler = BackgroundScheduler()
scheduler.start()
atexit.register(lambda: scheduler.shutdown()) # Shut down the scheduler when exiting the app


# Login security decorator
# ======================================================================

def login_required(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        # auth
        auth = request.authorization
        if not auth or auth.username!='xzhang1' or hashlib.sha1(auth.password).hexdigest()!='04a8e9fc933e4113ca82035de6a9fc22a9ac4765':
            print 'bad username/pswd detected: %s | %s' %(auth.username, hashlib.sha1(auth.password).hexdigest())
            return Response('Bad username/pswd', 403, {'WWW-Authenticate': 'Basic realm="Need login."'})
        else:
            return func(*args, **kwargs)
    return wrapped



# gui-sigma API: all sorts of shitty translations
# ======================================================================

def node_to_json(n):
    '''
    Recursively transform a node and all its printable attributes to a JSON object.
    Note that attributes are not involved, only the map attribute.
    '''
    new_json = {}
    # beautify
    # - use name for label and id
    # - use
    new_json['label']=new_json['id']=n.name
    new_json['color'] = shared.color_palette[n.moonphase()]
    # node_to_json
    for attr_name in vars(n):
        if attr_name != 'map' and attr_name in shared.attributes_printable:
            new_json[attr_name] = str( getattr(n,attr_name,None) )
        if attr_name == 'map':
            map_json = {'nodes': [], 'edges':[]}
            for src in n.map._dict:
                map_json['nodes'].append( node_to_json(src) )
                for dst in n.map._dict[src]:
                    map_json['edges'].append( {'id': id_generator(), 'source': src.name, 'target': dst.name, 'type' : 'arrow', 'color': shared.color_palette[min(src.moonphase(), dst.moonphase())] } )  # beautify
                for dst in n.map._dict2[src] if src in n.map._dict2 else []:
                    map_json['edges'].append( {'id': id_generator(), 'source': src.name, 'target': dst.name, 'type' : 'tapered', 'color': shared.color_palette[min(src.moonphase(), dst.moonphase())] } ) # type-II connection rendered as tapered
            new_json['map'] = map_json
    return new_json

def recursive_flatten_json(j, cur_prefix=None):
    '''
    returns a list of [
        cur eg. master.PbS QD.test,
        jam 'name phase property Pb S 44 46' for fuzzy matching sigma-json and nodes-json,
        [graphical properties to transfer]
    ]
    '''
    if cur_prefix:
        cur = cur_prefix + '.' + j['name']
    else:
        cur = j['name']
    result = []
    jam = ''
    attributes_sigmaonly = {}
    for key in j:
        # cur
        # jam
        if key in ['name','phase','property']:
            jam += ' ' + j[key]
        elif key == 'cell':
            jam += ' ' + j[key].splitlines()[5] + j[key].splitlines()[6]
        elif key in shared.attributes_printable:
            pass
        else:
            attributes_sigmaonly[key] = j[key]
    result.append([cur, jam, attributes_sigmaonly])

    if 'map' in j and 'nodes' in j['map']:
        for subj in j['map']['nodes']:
            result += recursive_flatten_json(subj, cur)
    return result

def lookup_cur_in_json(j, cur):
    if cur == 'master':
        return j    # note that the behavior is slightly different from qchem and sigmajs. the reason is that master is not global.
    elif '.' not in cur:
        l = [x for x in j['map']['nodes'] if x['name']==cur]
        if l:
            return l[0]
        else:
            raise KeyError('lookup_cur_in_json: cannot find %s under current node' %cur)
    else:
        return lookup_cur_in_json(lookup_cur_in_json(j,cur.split('.')[0]), '.'.join(cur.split('.')[1:]))

def combine_json(new_json, old_json=None):
    '''
    receives a json from sigma
    receives a new json converted from updated shared.nodes
    fuzzy inherit x,y (we don't want the graph to mess up every time we update shared.nodes)
    '''
    if old_json:
        for newj in recursive_flatten_json(new_json):
            if( any([oldj[0]==newj[0] for oldj in recursive_flatten_json(old_json)]) ):
                oldj = [oldj for oldj in recursive_flatten_json(old_json) if oldj[0]==newj[0]][0]
            else:
                best_jam = process.extractOne(newj[1], [oldj[1] for oldj in recursive_flatten_json(old_json)])
                if best_jam[1] > 50: # yeah, found match
                    oldj = [oldj for oldj in recursive_flatten_json(old_json) if oldj[1]==best_jam[0]][0]
                else:
                    oldj = [oldj[0], oldj[1], {}]
                if 'x' in oldj[2] and 'y' in oldj[2]:
                    oldj[2]['x'] = float(oldj[2]['x']) + random.uniform(0,0.1)
                    oldj[2]['y'] = float(oldj[2]['x']) + random.uniform(0,0.1)
            for key in oldj[2]:
                if oldj[2][key] and key not in newj[2]:
                    newj[2][key] = oldj[2][key]
                    lookup_cur_in_json(new_json,newj[0])[key] = oldj[2][key]
    for newj in recursive_flatten_json(new_json):
        tmp = lookup_cur_in_json(new_json, newj[0])
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

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# A lot of wrappers
# ======================================================================


@app.route('/dump_nodes', methods=['GET'])
@login_required
def dump_nodes():
    shared.save(shared.nodes, 'nodes')

# either load latest, or load a specific datetime_postfix.
@app.route('/load_nodes', methods=['GET','POST'])
@login_required
def load_nodes():
    shared.nodes = shared.load('nodes')

@app.route('/load_sigma', methods=['GET','POST'])
@login_required
def load_sigma():
    return jsonify(dynamic.load('sigma'))

@app.route('/dump_sigma', methods=['POST'])
@login_required
def dump_sigma():
    dynamic.save(request.get_json(force=True), 'sigma')

@app.route('/shutdown', methods=['GET', 'POST'])
@login_required
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    print 'Server shutting down... Thanks for using qchem.'

# for testing
@app.route('/hello_world')
def helloworld():
    return 'hello, world!'

@app.route('/request_', methods=['POST','GET'])
@login_required
def request_():  # either merge json, or use shared.nodes['master']     # yep, this is the magic function.
    if request.method == 'POST':
        old_json = request.get_json(force=True)
        if shared.DEBUG >= 2: print 'before node_to_json' + '*'*70
        new_json = node_to_json(engine.Map().lookup('master'))
        if shared.DEBUG >= 2: print 'after node_to_json' + '*'*70
        new_json = combine_json(new_json, old_json)
        return jsonify( new_json )

@app.route('/new_node', methods=['POST'])
@login_required
def new_node():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    n.map.add_node(graph.Node())

@app.route('/del_node', methods=['POST'])
@login_required
def del_node():
    j = request.get_json(force=True)
    node = engine.Map().lookup(j['cur']+'.'+j['name'])
    node.delete()

@app.route('/reset_node', methods=['POST'])
@login_required
def reset_node():
    j = request.get_json(force=True)
    engine.Map().lookup(j['cur']).reset()

@app.route('/duplicate_node', methods=['POST'])
@login_required
def duplicate_node():
    j = request.get_json(force=True)
    parent_node = engine.Map().lookup(j['cur'])
    n = parent_node.map.lookup(j['name'])
    parent_node.map.add_node(n.copy())

@app.route('/compute_node', methods=['POST'])
@login_required
def compute_node():
    j = request.get_json(force=True)
    engine.Map().lookup(j['cur']).compute(proposed_name=j['name'])  #delegate to parent, suggest compute name

@app.route('/setinterval_compute_node', methods=['POST'])
@login_required
def setinterval_compute_node():

    def setinterval_compute_node_base(j=request.get_json(force=True)):
        cur = j['cur'] + '.' + j['name']
        print 'setinterval job, computing    %s    , time %s' %(cur, str(datetime.now()))
        engine.Map().lookup(cur).compute()

    scheduler.add_job(
        func=setinterval_compute_node_base,
        trigger=IntervalTrigger(minutes=10),
        id='setinterval_compute_job',
        name='setinterval compute job',
        replace_existing=True)

@app.route('/stop_setinterval_compute_node', methods=['GET'])
@login_required
def stop_setinterval_compute_node():
    scheduler.remove_job('setinterval_compute_job')

@app.route('/get_text', methods=['POST'])
@login_required
def get_text():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    return jsonify({'text':str(n)})


@app.route('/edit_vars', methods=['POST'])
@login_required
def edit_vars():
    # Node() format input. Less powerful than edit, but more intuitive.
    # Updates mentioend and readable parts only.
    j = request.get_json(force=True)
    n = engine.Map().lookup(j.pop('cur'))
    for name,value in j.iteritems():
        if name == 'name' and ('.' in value or ',' in value or '=' in value):
            raise shared.CustomError('dot, comma and equal sign cannot be in name. ')
        if name not in shared.attributes_in:
            continue
        if getattr(engine, name.title(), None):
            value = getattr(engine, name.title())(value)
        setattr(n, name, value)

@app.route('/del_attr', methods=['POST'])
@login_required
def del_attr():
    # get cur and attr_name, and delete
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    delattr(n, j['attr_name'])

@app.route('/edit', methods=['POST'])
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
            shared.nodes[x.name] = x
    graph.Import(j['text'])
    #  update
    if node.name in shared.nodes:
        new_node = node.map.lookup(node.name)#pop
    else:
        raise shared.CustomError(node.__class__.__name__ + ': edit: You have not defined a same-name node (aka node with name %s which would have been read)' %(node.name))
    for varname in vars(new_node):
        setattr(node, varname, getattr(new_node, varname))


@app.route('/make_connection', methods=['GET'])
@login_required
def make_connection():
    if 'master' in shared.nodes and getattr(shared.nodes['master'], 'map', None):
        statuscolor = shared.color_palette[2]
    else:
        statuscolor = shared.color_palette[-1]
    return jsonify({'statuscolor':statuscolor, 'attributes_printable': shared.attributes_printable, 'attributes_in': shared.attributes_in, 'DEBUG': shared.DEBUG})


@app.route('/cut_ref', methods=['POST'])
@login_required
def cut_ref():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur']+'.'+j['name'])
    p = engine.Map().lookup(j['cur'])
    shared.nodes[n.name] = n
    p.map.del_node(n)

@app.route('/paste_ref', methods=['POST'])
@login_required
def paste_ref():
    j = request.get_json(force=True)
    p = engine.Map().lookup(j['cur'])
    l = [n for n in shared.nodes.values() if n!=engine.Map().lookup('master')]
    if not l:
        raise shared.CustomError('paste_ref error: shared.nodes only contains master. nothing pastable')
    if len(l) > 1:
        print 'paste_ref warning: more than one nodes pastable. pastable nodes are [%s]' %([n.name for n in l])
    n = l[0]
    print 'paste_ref: adding node {%s}' %n.name
    p.map.add_node(n)
    shared.nodes.pop(n.name)


@app.route('/add_edge', methods=['POST'])
@login_required
def add_edge():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    n.map.add_edge(j['src'],j['dst'])

@app.route('/del_edge', methods=['POST'])
@login_required
def del_edge():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    n.map.del_edge(j['src'],j['dst'])

@app.route('/copy_remote_folder_name', methods=['POST'])
@login_required
def copy_remote_folder_name():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    try:
        return jsonify({'remote_folder_name': n.vasp.remote_folder_name})
    except:
        return jsonify({'remote_folder_name':'remote_folder_name_not_assigned'})

@app.route('/copy_path', methods=['POST'])
@login_required
def copy_path():
    j = request.get_json(force=True)
    n = engine.Map().lookup(j['cur'])
    try:
        return jsonify({'path': n.path})
    except:
        return jsonify({'path':'path_not_assigned'})


# Run flask
# ======================================================================

def flaskThread():
    app.run(host='0.0.0.0',port=5000, debug = False,
                ssl_context=('/home/xzhang1/.openssl/mycert.pem','/home/xzhang1/.openssl/mykey.pem'))

def GUI(safe=True):
    print "Note: Flask dies hard. If 'another program is using your port', you need to pkill -9 python. Needless to say, dangerous."
    if safe:
        print "Safe mode: only one instance can access data at a time, avoiding conflicts. "
        print "Consequently, you may not run any command while GUI is running."
        print "Alternatively, open another session and read/write from files. It's safe, but you're not sharing vars."
        flaskThread()
    else:
        print "Race mode. Both this session and GUI will access same set of variables. Warning: race / conflicts possible."
        thread.start_new_thread(flaskThread, ())

if __name__ == "__main__":
    GUI()
    # app.run(host='0.0.0.0',port=5000, debug = False, ssl_context=('/home/xzhang1/.openssl/mycert.pem','/home/xzhang1/.openssl/mykey.pem'))
    # app.run(host='0.0.0.0', port=5000, debug=False)
