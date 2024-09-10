from flask import Flask, render_template, request, jsonify
from configparser import ConfigParser
import os
from rdflib import ConjunctiveGraph, Namespace, Literal
import traceback
from r2rml_test_cases.test import test_one, generate_results, database_load
from database_manager import DatabaseManager

app = Flask(__name__)

# Load configuration
config = ConfigParser()
config.read('config.ini')

TEST_CASES_DIR = os.path.join(os.path.dirname(__file__), 'r2rml_test_cases')

RDB2RDFTEST = Namespace("http://purl.org/NET/rdb2rdf-test#")
TESTDEC = Namespace("http://www.w3.org/2006/03/test-description#")
DCELEMENTS = Namespace("http://purl.org/dc/terms/")

# Load manifest graph
manifest_graph = ConjunctiveGraph()
manifest_graph.parse(os.path.join(TEST_CASES_DIR, "manifest.ttl"), format='turtle')

db_manager = DatabaseManager()

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
    
    return run_single_test(test_id, database_system)

@app.route('/run_all_tests', methods=['POST'])
def run_all_tests():
    database_system = request.form['database_system']
    tests = [f for f in os.listdir(TEST_CASES_DIR) if os.path.isdir(os.path.join(TEST_CASES_DIR, f)) and f.startswith('R2RMLTC')]
    
    all_results = []
    for test_id in tests:
        result = run_single_test(test_id, database_system)
        all_results.append(result)
    
    return jsonify({'status': 'success', 'results': all_results})

def run_single_test(test_id, database_system):
    test_dir = os.path.join(TEST_CASES_DIR)
    os.chdir(test_dir)

    try:
        container = db_manager.get_container(database_system)
        
        # Reset the database for the new test
        db_manager.reset_database(database_system)
        
        # Load the specific test database
        test_uri = manifest_graph.value(subject=None, predicate=DCELEMENTS.identifier, object=Literal(test_id))
        database_uri = manifest_graph.value(subject=test_uri, predicate=RDB2RDFTEST.database, object=None)
        database = manifest_graph.value(subject=database_uri, predicate=RDB2RDFTEST.sqlScriptFile, object=None)
        database_load(database, database_system)
        
        results = test_one(test_id, database_system, config, manifest_graph)
        generate_results(database_system, config, results)
        
        os.chdir(os.path.dirname(__file__))
        
        return {'status': 'success', 'test_id': test_id, 'results': results}
    except Exception as e:
        error_traceback = traceback.format_exc()
        os.chdir(os.path.dirname(__file__))
        return {
            'status': 'error',
            'test_id': test_id,
            'message': str(e),
            'traceback': error_traceback
        }

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