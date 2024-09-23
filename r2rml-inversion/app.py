from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from configparser import ConfigParser
import os
from rdflib import ConjunctiveGraph, Namespace, Literal
import traceback
from r2rml_test_cases.test import test_one, generate_results, database_load
from database_manager import DatabaseManager
import json
import threading
import base64
from datetime import date, datetime
import math
from poc_inversion import inversion
from morph_kgc.mapping.mapping_parser import retrieve_mappings
from morph_kgc.args_parser import load_config_from_argument


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        elif isinstance(obj, float):
            if math.isnan(obj):
                return "NaN"
            elif math.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
        elif hasattr(obj, '__dict__'):
            return str(obj)
        try:
            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return list(iterable)
        return super().default(obj)


app = Flask(__name__)

# Load configuration
config = ConfigParser()
config.read('config.ini')

TEST_CASES_DIR = os.path.join(os.path.dirname(__file__), 'r2rml_test_cases')

MORPH_KCG_CONFIG_FILEPATH = os.path.join(os.path.dirname(__file__), 'morph_kgc_config.ini')

RDB2RDFTEST = Namespace("http://purl.org/NET/rdb2rdf-test#")
TESTDEC = Namespace("http://www.w3.org/2006/03/test-description#")
DCELEMENTS = Namespace("http://purl.org/dc/terms/")

# Load manifest graph
manifest_graph = ConjunctiveGraph()
manifest_graph.parse(os.path.join(TEST_CASES_DIR, "manifest.ttl"), format='turtle')

db_manager = DatabaseManager()
db_manager.get_container('graphdb')

# Add a flag to track if tests are running
tests_running = threading.Event()
cancel_tests = threading.Event()

def get_mapping_filename(test_id):
    letter: str = test_id[-1].lower()
    return f'r2rml{letter}.ttl' if letter.isalpha() else 'r2rml.ttl'

def sanitize_data(data):
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data):
            return "NaN"
        elif math.isinf(data):
            return "Infinity" if data > 0 else "-Infinity"
    elif isinstance(data, (int, str, bool, type(None))):
        return data
    else:
        return str(data)

@app.route('/')
def index():
    tests = sorted([f for f in os.listdir(TEST_CASES_DIR) if os.path.isdir(os.path.join(TEST_CASES_DIR, f)) and f.startswith('R2RMLTC')])
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

    try:
        sanitized_result = sanitize_data(result)
        json_result = json.dumps(sanitized_result, cls=CustomJSONEncoder)
        return app.response_class(
            response=json_result,
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        error_msg = f"Error serializing result for test {test_id}: {str(e)}"
        return jsonify({
            'status': 'error',
            'test_id': test_id,
            'message': error_msg
        }), 500

@app.route('/run_all_tests', methods=['GET'])
def run_all_tests():
    if tests_running.is_set():
        cancel_tests.set()
        tests_running.clear()
    
    database_system = request.args.get('database_system')
    tests = sorted([f for f in os.listdir(TEST_CASES_DIR) if os.path.isdir(os.path.join(TEST_CASES_DIR, f)) and f.startswith('R2RMLTC')])
    
    cancel_tests.clear()
    tests_running.set()
    
    def generate():
        for test_id in tests:
            if cancel_tests.is_set():
                yield f"data: {json.dumps({'status': 'cancelled', 'message': 'Tests cancelled by user'})}\n\n"
                break
            result = run_single_test(test_id, database_system)
            try:
                sanitized_result = sanitize_data(result)
                json_result = json.dumps(sanitized_result, cls=CustomJSONEncoder)
                yield f"data: {json_result}\n\n"
            except Exception as e:
                error_msg = f"Error serializing result for test {test_id}: {str(e)}"
                yield f"data: {json.dumps({'status': 'error', 'test_id': test_id, 'message': error_msg})}\n\n"
        tests_running.clear()
        yield "event: complete\ndata: All tests completed\n\n"
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/get_file_content', methods=['GET'])
def get_file_content():
    test_id = request.args.get('test_id')
    file_type = request.args.get('type')  # 'expected' o 'actual'
    database_system = request.args.get('database_system')
    
    if file_type == 'expected':
        file_path = os.path.join(TEST_CASES_DIR, test_id, 'output.ttl')
    elif file_type == 'actual':
        file_path = os.path.join(TEST_CASES_DIR, test_id, f'engine_output-{database_system}.ttl')
    else:
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return jsonify({'content': content})
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

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
        database_load(database, database_system)
        
        # Get database structure
        db_content = db_manager.get_database_content(database_system)
        
        # Get mapping content
        mapping_filename = get_mapping_filename(test_id)
        mapping_file = os.path.join(TEST_CASES_DIR, test_id, mapping_filename)
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_content = f.read()
        
        # Get the purpose of the test
        purpose = manifest_graph.value(subject=test_uri, predicate=TESTDEC.purpose, object=None)
        purpose = purpose.toPython() if purpose else "Purpose not specified"
        
        raw_results = test_one(test_id, database_system, config, manifest_graph)        
                
        inversion_result = inversion(MORPH_KCG_CONFIG_FILEPATH, test_id)

        processed_results = process_results(raw_results, db_content, mapping_content, test_id, database_system, config, purpose, inversion_result)
        generate_results(database_system, config, raw_results)
        
        os.chdir(os.path.dirname(__file__))
        
        return {
            'status': 'success', 
            'test_id': test_id, 
            'results': processed_results
        }
    except Exception as e:
        error_traceback = traceback.format_exc()
        os.chdir(os.path.dirname(__file__))
        return {
            'status': 'error',
            'test_id': test_id,
            'message': str(e),
            'traceback': error_traceback
        }

def process_results(raw_results, db_content, mapping_content, test_id, database_system, config, purpose, inversion_result):
    processed_results = {
        'headers': ['Test ID', 'Purpose', 'Result', 'Inversion Query'],
        'data': []
    }
    
    for row in raw_results[1:]:  # Skip the header row
        expected_content, actual_content = get_file_contents(test_id, database_system, config)

        formatted_queries = []
        for source, query in inversion_result.items():
            formatted_queries.append(query.strip())
        formatted_inversion_result = "\n\n".join(formatted_queries)

        processed_row = {
            'testid': row[3] if len(row) > 3 else 'N/A',
            'purpose': purpose,
            'result': row[4] if len(row) > 4 else 'N/A',
            'expected_result': expected_content,
            'actual_result': actual_content,
            'db_content': db_content,
            'mapping': mapping_content,
            'inversion_query': formatted_inversion_result
        }
        processed_results['data'].append(processed_row)
    
    return processed_results

def get_file_contents(test_id, database_system, config: ConfigParser):
    output_format = config['properties'].get('output_format', 'ntriples')
    ext = 'ttl' if output_format == 'turtle' else 'nt' if output_format == 'ntriples' else 'nq'
    
    # Get the last character of the test_id
    last_char: str = test_id[-1]
    
    # Determine the suffix for the expected file
    suffix = last_char.lower() if last_char.isalpha() else ''
    
    expected_file = os.path.join(TEST_CASES_DIR, test_id, f'mapped{suffix}.nq')
    actual_file = os.path.join(TEST_CASES_DIR, test_id, f'engine_output-{database_system}.{ext}')

    expected_content = read_file_content(expected_file)
    actual_content = read_file_content(actual_file)
    
    return expected_content, actual_content

def read_file_content(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return "File not found"

def process_db_content(db_content):
    processed_content = {}
    for table_name, table_data in db_content.items():
        if isinstance(table_data, str):
            # Se table_data è una stringa, è probabilmente un messaggio di errore
            processed_content[table_name] = {
                'error': table_data,
                'columns': [],
                'data': []
            }
        else:
            # Assumiamo che table_data sia un dizionario con 'columns' e 'data'
            processed_table = {
                'columns': table_data.get('columns', []),
                'data': []
            }
            for row in table_data.get('data', []):
                processed_row = []
                for value in row:
                    if isinstance(value, memoryview):
                        # Aggiungiamo un prefisso per indicare che si tratta di un'immagine PNG
                        processed_row.append("data:image/png;base64," + base64.b64encode(value.tobytes()).decode('utf-8'))
                    else:
                        processed_row.append(value)
                processed_table['data'].append(processed_row)
            processed_content[table_name] = processed_table
    return processed_content


if __name__ == '__main__':
    app.run(debug=True)