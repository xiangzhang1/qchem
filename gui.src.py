from qchem import *
from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/load', methods=['POST', 'GET'])
def load():
    Load()
    return 'load'

@app.route('/dump', methods=['POST', 'GET'])
def dump():
    Dump()
    return 'dump'

@app.route('/test', methods=['POST', 'GET'])
def test():
    with open('test.md') as f:
        Import(f.read())
    return 'test'
