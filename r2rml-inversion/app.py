from flask import Flask, render_template, request, jsonify
from configparser import ConfigParser
import os
import csv
from rdflib import ConjunctiveGraph, Namespace
import traceback
from r2rml_test_cases.test import test_one, database_up, database_down, generate_results, merge_results

app = Flask(__name__)

# Load configuration
config = ConfigParser()
config.read('config.ini')

TEST_CASES_DIR = os.path.join(os.path.dirname(__file__), 'r2rml_test_cases')

# Load manifest graph
manifest_graph = ConjunctiveGraph()
manifest_graph.parse(os.path.join(TEST_CASES_DIR, "manifest.ttl"), format='turtle')


def get_mapping_filename(test_id):
    letter = test_id[-1].lower()
    return f'r2rml{letter}.ttl'

@app.route('/')
def index():
    tests = [f for f in os.listdir(TEST_CASES_DIR) if os.path.isdir(os.path.join(TEST_CASES_DIR, f)) and f.startswith('R2RMLTC')]
    return render_template('index.jinja', tests=tests)

@app.route('/run_test', methods=['POST'])
def run_test():
    test_id = request.form['test_id']
    database_system = request.form['database_system']
    
    config['properties']['database_system'] = database_system
    config['properties']['tests'] = test_id
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    test_dir = os.path.join(TEST_CASES_DIR)
    os.chdir(test_dir)

    try:
        database_up(database_system)
        results = test_one(test_id, database_system, config, manifest_graph)
        generate_results(database_system, config, results)
        database_down(database_system)
        merge_results()
        
        with open(os.path.join(TEST_CASES_DIR, f'results-{database_system}.csv'), 'r', encoding='utf8') as f:
            reader = csv.reader(f)
            results = list(reader)
        
        os.chdir(os.path.dirname(__file__))
        
        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        error_traceback = traceback.format_exc()
        os.chdir(os.path.dirname(__file__))
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': error_traceback
        })

@app.route('/get_mapping/<test_id>')
def get_mapping(test_id):
    mapping_filename = get_mapping_filename(test_id)
    mapping_file = os.path.join(TEST_CASES_DIR, test_id, mapping_filename)
    
    if not os.path.exists(mapping_file):
        return jsonify({'status': 'error', 'message': f"Mapping file not found: {mapping_filename}"})
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_content = f.read()
    return jsonify({'status': 'success', 'content': mapping_content})

if __name__ == '__main__':
    app.run(debug=True)