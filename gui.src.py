from qchem import *
from flask import Flask
from flask_cors import CORS, cross_origin
from cStringIO import StringIO
import sys

app = Flask(__name__)
CORS(app)

@app.route('/load', methods=['POST', 'GET'])
def load():
    Load()
    return 'gui: load() success'

@app.route('/dump', methods=['POST', 'GET'])
def dump():
    Dump()
    return 'gui: dump() success'

@app.route('/test', methods=['POST', 'GET'])
def test():
    sys.stdout = mystdout = StringIO()
    try:
        with open('test.md') as f:
            Import(f.read())
        return mystdout.getvalue() + '\n' + '-'*30 + '\ngui: test() success'
    except Exception as e:
        return mystdout.getvalue() + '\n' + str(e) + '\n' + '-'*30 + '\ngui: test() failed as above. game over.' 
    sys.stdout  = sys.__stdout__

