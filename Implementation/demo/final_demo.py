from flask import Flask, json, render_template, request, send_from_directory, send_file
import rdflib
from turbo_flask import Turbo

from tempfile import TemporaryDirectory 
import pathlib
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "pyrdf4j"))
from poc_inversion import inversion

import morph_kgc

import shutil

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
        
@app.route('/')
def index():
    presets = [p.name for p in (pathlib.Path(app.root_path) / 'static' / 'presets').iterdir() if p.is_dir()]
    return send_from_directory(os.path.join(app.root_path, 'static', 'html'), 'inversion.html') # we really dont use any flask specific functions anymore... this demo be served with a generic web server and an API

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/get_presets', methods=['GET'])
def get_presets():
    presets = [p.name for p in (pathlib.Path(app.root_path) / 'static' / 'presets').iterdir() if p.is_dir()]
    return presets

@app.route('/get_preset', methods=['POST'])
def get_preset():
    preset = request.form['preset']
    preset_folder = pathlib.Path(app.root_path) / 'static' / 'presets' / preset
    with open(preset_folder / 'config.yml') as f:
        config = yaml.safe_load(f)
        sources_names = config['sources']
        sources_dict = {source_name: (preset_folder / source_name).read_text() for source_name in sources_names}
        yarrml_mappings = (preset_folder / config['yarrrml']).read_text()
    return {'sources': sources_dict, 'yarrrml': yarrml_mappings}

def translate_mappings(yarrrml:str):
    input_data_raw = yaml.safe_load(yarrrml)
    externals = input_data_raw.get("external")
    if externals:
        for key in externals.keys():
            match_string = f"$(_{key})"
            data = externals.get(key)
            yarrrml = yarrrml.replace(match_string, data)
    with PrintHider():
        rml_mappings = yatter.translate(yaml.safe_load(yarrrml))
    return rml_mappings

@app.route('/generate_knowledge_graph', methods=['POST'])
def generate_knowledge_graph():
    sources = json.loads(request.form['sources']) 
    yarrrml_mapping = request.form['yarrrml']
    mapping = translate_mappings(yarrrml_mapping)
    current_dir = pathlib.Path(os.getcwd())
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        for source_name, source_content in sources.items():
            with open(source_name, 'w', encoding="utf-8", newline="") as f:
                f.write(source_content)
        with open('mapping.ttl', 'w', encoding="utf-8", newline="") as f:
            f.write(mapping)
        with PrintHider():
            try:
                graph:rdflib.Graph = morph_kgc.materialize(MORPH_CONFIG)
            except FileNotFoundError as e:
                graph = rdflib.Graph()
                pass
        print(graph)
        subjects_count = len(set(graph.subjects()))
        triples_count = len(graph)
        subjects_count = subjects_count
        triples_count = triples_count
        graph.serialize('output.nq', format='ntriples')
        with open('output.nq', 'r', encoding="utf-8", newline="") as f:
            knowledge_graph = f.read()
        graph = knowledge_graph
        os.chdir(current_dir)
    return {'graph': graph, 'subject_count': subjects_count, 'triple_count': triples_count}

@app.route('/invert_knowledge_graph', methods=['POST'])
def invert_knowledge_graph():
    knowledge_graph = request.form['knowledge_graph']
    yarrrml_mapping = request.form['yarrrml']
    mapping = translate_mappings(yarrrml_mapping)
    current_dir = pathlib.Path(os.getcwd())
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        with open('output.nq', 'w', encoding="utf-8", newline="") as f:
            f.write(knowledge_graph.strip() + " " * 10000)
        with open('mapping.ttl', 'w', encoding="utf-8", newline="") as f:
            f.write(mapping)
        try:
            generated_sources = inversion(MORPH_CONFIG, "demo")
        except Exception as e:
            generated_sources = {}
            pass
        generated_sources_formatted = {source_name: format_if_json(source_content) for source_name, source_content in generated_sources.items()}
        os.chdir(current_dir)
    return {'sources': generated_sources_formatted}

def format_if_json(data):
    try:
        return json.dumps(json.loads(data), indent=2)
    except:
        return data

if __name__ == '__main__':
    app.run(debug=True)