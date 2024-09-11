from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from configparser import ConfigParser
import os
from rdflib import ConjunctiveGraph, Namespace, Literal
import traceback
from r2rml_test_cases.test import test_one, generate_results, database_load
from database_manager import DatabaseManager
import json
import threading

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

# Add a flag to track if tests are running
tests_running = threading.Event()
cancel_tests = threading.Event()

def get_mapping_filename(test_id):
    letter = test_id[-1].lower()
    return f'r2rml{letter}.ttl'

@app.route('/')
def index():
    tests = [f for f in os.listdir(TEST_CASES_DIR) if os.path.isdir(os.path.join(TEST_CASES_DIR, f)) and f.startswith('R2RMLTC')]
    return render_template('index.jinja', tests=tests)

@app.route('/run_test', methods=['POST'])
def run_test():
    if tests_running.is_set():
        cancel_tests.set()
        tests_running.clear()
    
    test_id = request.form['test_id']
    database_system = request.form['database_system']
    
    cancel_tests.clear()
    tests_running.set()
    result = run_single_test(test_id, database_system)
    tests_running.clear()
    return jsonify(result)

@app.route('/run_all_tests', methods=['GET'])
def run_all_tests():
    if tests_running.is_set():
        cancel_tests.set()
        tests_running.clear()
    
    database_system = request.args.get('database_system')
    tests = [f for f in os.listdir(TEST_CASES_DIR) if os.path.isdir(os.path.join(TEST_CASES_DIR, f)) and f.startswith('R2RMLTC')]
    
    cancel_tests.clear()
    tests_running.set()
    
    def generate():
        for test_id in tests:
            if cancel_tests.is_set():
                yield f"data: {json.dumps({'status': 'cancelled', 'message': 'Tests cancelled by user'})}\n\n"
                break
            result = run_single_test(test_id, database_system)
            yield f"data: {json.dumps(result)}\n\n"
        tests_running.clear()
        yield "event: complete\ndata: All tests completed\n\n"
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')

def run_single_test(test_id, database_system):
    test_dir = os.path.join(TEST_CASES_DIR)
    os.chdir(test_dir)

    try:
        if cancel_tests.is_set():
            return {'status': 'cancelled', 'test_id': test_id, 'message': 'Test cancelled by user'}
                
        # Reset the database for the new test
        db_manager.reset_database(database_system)
        
        # Load the specific test database
        test_uri = manifest_graph.value(subject=None, predicate=DCELEMENTS.identifier, object=Literal(test_id))
        database_uri = manifest_graph.value(subject=test_uri, predicate=RDB2RDFTEST.database, object=None)
        database = manifest_graph.value(subject=database_uri, predicate=RDB2RDFTEST.sqlScriptFile, object=None)
        
        if database is None:
            raise ValueError(f"No database script found for test {test_id}")
        
        database_load(database.toPython(), database_system)
        
        raw_results = test_one(test_id, database_system, config, manifest_graph)
        
        # Process the results to match the desired structure
        processed_results = process_results(raw_results)
        
        generate_results(database_system, config, raw_results)  # Keep this for file generation if needed
        
        os.chdir(os.path.dirname(__file__))
        
        return {'status': 'success', 'test_id': test_id, 'results': processed_results}
    except Exception as e:
        error_traceback = traceback.format_exc()
        os.chdir(os.path.dirname(__file__))
        return {
            'status': 'error',
            'test_id': test_id,
            'message': str(e),
            'traceback': error_traceback
        }

def process_results(raw_results):
    # Skip the header row and process each result row
    processed = []
    for row in raw_results[1:]:
        processed.append({
            'test_id': row[3],  # Assuming test_id is at index 3
            'platform': row[1],
            'rdbms': row[2],
            'result': row[4]
        })
    return processed

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