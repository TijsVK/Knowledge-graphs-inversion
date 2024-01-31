from flask import Flask, render_template, request, send_from_directory
import rdflib
from turbo_flask import Turbo

import pathlib

from ..poc_inversion import inversion

import morph_kgc

import shutil

import sys, os

import time

import pretty_yarrrml2rml
import yatter
import yaml

app = Flask(__name__)
turbo = Turbo(app)

MORPH_CONFIG = f"""
    [CONFIGURATION]
    # INPUT
    na_values=,#N/A,N/A,#N/A N/A,n/a,NA,<NA>,#NA,NULL,null,NaN,nan,None

    # OUTPUT
    output_file=output.nq
    output_dir=
    output_format=N-TRIPLES
    only_printable_characters=no
    safe_percent_encoding=

    # MAPPINGS
    mapping_partitioning=PARTIAL-AGGREGATIONS
    infer_sql_datatypes=no

    # MULTIPROCESSING
    number_of_processes=

    # LOGS
    logging_level=WARNING
    logs_file=


    [DataSource1]
    mappings: mapping.ttl
"""

class PrintHider:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def push_yarrml(yarrml):
    with app.app_context():
        turbo.push(turbo.replace(render_template('yarrrml_textarea.html', Yarrrml=yarrml), 'yarrrml'))

def push_generated_mappings(rml_content):
    with app.app_context():
        turbo.push(turbo.replace(render_template('generated_mappings_textarea.html', Mappings=rml_content), 'rml_mappings'))

def push_source_name(source_name):
    with app.app_context():
        turbo.push(turbo.replace(render_template('source_name_text_box.html', SourceName=source_name), 'source_name'))

def push_source_content(source_content):
    with app.app_context():
        turbo.push(turbo.replace(render_template('source_content_textarea.html', SourceContent=source_content), 'source_content'))

def push_knowledge_graph(knowledge_graph):
    with app.app_context():
        turbo.push(turbo.replace(render_template('knowledge_graph_textarea.html', KnowledgeGraph=knowledge_graph), 'knowledge_graph'))

def push_knowledge_graph_stats(subjects_count, triples_count):
    with app.app_context():
        turbo.push(turbo.replace(render_template('knowledge_graph_stats.html', SubjectsCount=subjects_count, TriplesCount=triples_count), 'knowledge_graph_stats'))

def push_inverted_source(inverted_source_contents):
    with app.app_context():
        turbo.push(turbo.replace(render_template('inverted_source_textarea.html', InvertedSourceContents=inverted_source_contents), 'inverted_source'))

@app.route('/')
def index():
    presets = [p.name for p in (pathlib.Path(app.root_path) / 'static' / 'presets').iterdir() if p.is_dir()]
    return render_template('demo.html', sources= presets)

@app.route('/update', methods=['POST'])
def update():
    print(request.get_data())
    # print(request.form['source'])
    return ''

@app.route('/generate_mappings', methods=['POST'])
def generate_mappings():
    yarrrml = request.form['yarrrml']
    generate_and_push_mappings(yarrrml)
    return ''

def generate_and_push_mappings(yarrrml):
    with PrintHider():
        rml_content = yatter.translate(yaml.safe_load(yarrrml))
    push_generated_mappings(rml_content)


@app.route('/load_preset', methods=['POST'])
def load_preset():
    preset = request.form['preset']
    if preset == 'None':
        push_yarrml("")
        push_generated_mappings("")
        push_source_name("")
        push_source_content("")
        push_knowledge_graph("")
        push_knowledge_graph_stats(0, 0)
        return ''
    preset_dir = pathlib.Path(app.root_path) / 'static' / 'presets' / preset
    with open(preset_dir / 'yarrrml.yml') as f:
        yarrrml = f.read()
    push_yarrml(yarrrml)
    generate_and_push_mappings(yarrrml)
    # get all files with .csv extension
    csv_files = [f for f in preset_dir.glob('*.csv')]
    csv_file = csv_files[0]
    filename = csv_file.name
    push_source_name(filename)
    file_contents = csv_file.read_bytes().decode('utf-8')
    push_source_content(file_contents)
    return ''

@app.route('/generate_knowledge_graph', methods=['POST'])
def generate_knowledge_graph():
    source_name = request.form['source_name']
    source_content = request.form['source_content']
    mapping = request.form['rml_mappings']
    current_dir = pathlib.Path(os.getcwd())
    tmp_dir = pathlib.Path(app.root_path) / 'tmp'
    tmp_dir.mkdir(exist_ok=True)
    os.chdir(tmp_dir)
    with open(source_name, 'w', encoding="utf-8", newline="") as f:
        f.write(source_content)
    with open('mapping.ttl', 'w', encoding="utf-8", newline="") as f:
        f.write(mapping)
    with PrintHider():
        graph:rdflib.Graph = morph_kgc.materialize(MORPH_CONFIG)
    print(graph)
    subjects_count = len(set(graph.subjects()))
    triples_count = len(graph)
    push_knowledge_graph_stats(subjects_count, triples_count)
    graph.serialize('output.nq', format='ntriples')
    with open('output.nq', 'r', encoding="utf-8", newline="") as f:
        knowledge_graph = f.read()
    push_knowledge_graph(knowledge_graph)
    os.chdir(current_dir)
    shutil.rmtree(tmp_dir)
    return ''

@app.route('/invert_knowledge_graph', methods=['POST'])
def invert_knowledge_graph():
    knowledge_graph = request.form['knowledge_graph']
    mapping = request.form['rml_mappings']
    current_dir = pathlib.Path(os.getcwd())
    tmp_dir = pathlib.Path(app.root_path) / 'tmp'
    tmp_dir.mkdir(exist_ok=True)
    os.chdir(tmp_dir)
    with open('output.nq', 'w', encoding="utf-8", newline="") as f:
        f.write(knowledge_graph.strip() + " " * 10000)
    with open('mapping.ttl', 'w', encoding="utf-8", newline="") as f:
        f.write(mapping)
    inversion(MORPH_CONFIG)
    csv_files = [f for f in tmp_dir.glob('*.csv')]
    csv_file = csv_files[0]
    file_contents = csv_file.read_bytes().decode('utf-8')
    push_inverted_source(file_contents)
    os.chdir(current_dir)
    shutil.rmtree(tmp_dir)
    return ''
    

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.run(debug=True)