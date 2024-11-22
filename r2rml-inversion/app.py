import json
import math
import os
import traceback
from configparser import ConfigParser
from datetime import date, datetime

import pandas as pd
import sqlalchemy
from flask import (Flask, Response, jsonify, render_template, request,
                   stream_with_context)
from rdflib import ConjunctiveGraph, Literal, Namespace

from database_manager import DatabaseManager
from poc_inversion import inversion
from r2rml_test_cases.test import database_load, generate_results, test_one


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

DEST_DB_SYSTEM = 'dest_postgresql'

# Load manifest graph
manifest_graph = ConjunctiveGraph()
manifest_graph.parse(os.path.join(TEST_CASES_DIR, "manifest.ttl"), format='turtle')

db_manager = DatabaseManager()
db_manager.get_container(DEST_DB_SYSTEM)
db_manager.get_container('postgresql')

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
    test_id = request.form['test_id']
    database_system = request.form['database_system']
    
    result = run_single_test(test_id, database_system)

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
    database_system = request.args.get('database_system')
    tests = sorted([f for f in os.listdir(TEST_CASES_DIR) if os.path.isdir(os.path.join(TEST_CASES_DIR, f)) and f.startswith('R2RMLTC')])
        
    def generate():
        for test_id in tests:
            result = run_single_test(test_id, database_system)
            try:
                sanitized_result = sanitize_data(result)
                json_result = json.dumps(sanitized_result, cls=CustomJSONEncoder)
                yield f"data: {json_result}\n\n"
            except Exception as e:
                error_msg = f"Error serializing result for test {test_id}: {str(e)}"
                yield f"data: {json.dumps({'status': 'error', 'test_id': test_id, 'message': error_msg})}\n\n"
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

def drop_tables(db_manager: DatabaseManager, database_system):
    try:
        connection_string = db_manager.get_connection_string(database_system)
        engine = db_manager.create_engine(connection_string)
        with engine.begin() as connection:
            # Get the metadata
            metadata = sqlalchemy.MetaData()
            metadata.reflect(bind=engine)
            
            # Drop all tables
            metadata.drop_all(engine)
            
        print(f"All tables dropped for {database_system}")
    except Exception as e:
        print(f"Error dropping tables for {database_system}: {str(e)}")
    finally:
        engine.dispose()

def run_single_test(test_id, database_system):
    test_dir = os.path.join(TEST_CASES_DIR)
    os.chdir(test_dir)

    try:
        # Reset databases for the new test
        drop_tables(db_manager, database_system)
        drop_tables(db_manager, DEST_DB_SYSTEM)

        # Load test-specific data
        test_uri = manifest_graph.value(subject=None, predicate=DCELEMENTS.identifier, object=Literal(test_id))
        database_uri = manifest_graph.value(subject=test_uri, predicate=RDB2RDFTEST.database, object=None)
        database = manifest_graph.value(subject=database_uri, predicate=RDB2RDFTEST.sqlScriptFile, object=None)
        
        # Load the database for the test
        database_load(database, database_system)

        # Get mapping content
        mapping_filename = get_mapping_filename(test_id)
        mapping_file = os.path.join(TEST_CASES_DIR, test_id, mapping_filename)
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_content = f.read()
        
        # Get the purpose of the test
        purpose = manifest_graph.value(subject=test_uri, predicate=TESTDEC.purpose, object=None)
        purpose = purpose.toPython() if purpose else "Purpose not specified"
        
        # Run the R2RML test
        raw_results = test_one(test_id, database_system, config, manifest_graph)        
        
        # Perform inversion
        dest_db_url = db_manager.get_connection_string(DEST_DB_SYSTEM)
        inversion_result = inversion(MORPH_KCG_CONFIG_FILEPATH, test_id, dest_db_url)

        inversion_success = bool(inversion_result)

        # Compare original and inverted tables
        if inversion_success:
            databases_equal, comparison_message, source_content, dest_content = compare_databases(db_manager, database_system, DEST_DB_SYSTEM)
        else:
            databases_equal = True
            comparison_message = "Inversion failed or was skipped, database comparison not performed."
            source_content = None
            dest_content = None

        # Process and generate results
        processed_results = process_results(
            raw_results, mapping_content, test_id, database_system, 
            config, purpose, inversion_result, databases_equal, comparison_message,
            source_content, dest_content
        )
        generate_results(database_system, config, raw_results)
        
        # Return to original directory
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

def process_results(raw_results, mapping_content, test_id, database_system, config, purpose, inversion_result, 
                    databases_equal, comparison_message, source_content, dest_content):
    processed_results = {
        'headers': ['Test ID', 'Purpose', 'Result', 'Expected Result', 'Actual Result', 'Mapping', 'SPARQL Query', 'Inversion Query', 'Inversion Success', 'Tables Comparison'],
        'data': []
    }
    
    for row in raw_results[1:]:  # Skip the header row
        expected_content, actual_content = get_file_contents(test_id, database_system, config)

        formatted_queries = []
        sparql_queries = []
        for source, result in inversion_result.items():
            formatted_queries.append(result['inverted_query'].strip())
            sparql_queries.append(result['sparql_query'])
                
        formatted_inversion_result = "\n\n".join(formatted_queries)
        formatted_sparql_queries = "\n\n".join(filter(None, sparql_queries))

        processed_row = {
            'testid': row[3] if len(row) > 3 else 'N/A',
            'purpose': purpose,
            'result': row[4] if len(row) > 4 else 'N/A',
            'expected_result': expected_content,
            'actual_result': actual_content,
            'mapping': mapping_content,
            'sparql_query': formatted_sparql_queries,
            'inversion_query': formatted_inversion_result,
            'inversion_success': databases_equal,
            'tables_equal': databases_equal,
            'comparison_message': comparison_message,
            'original_tables': source_content,
            'inverted_tables': dest_content
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

def compare_databases(db_manager, source_system, dest_system):
    try:
        source_content = db_manager.get_database_content(source_system)
        dest_content = db_manager.get_database_content(dest_system)

        if not source_content or not dest_content:
            return False, "One or both databases are empty or couldn't be accessed", None, None

        if set(source_content.keys()) != set(dest_content.keys()):
            return False, "Tables in source and destination databases do not match", source_content, dest_content

        mismatched_tables = []
        for table_name in source_content.keys():
            source_table = source_content[table_name]
            dest_table = dest_content[table_name]
            
            if set(source_table['columns']) != set(dest_table['columns']):
                mismatched_tables.append(f"{table_name} (columns mismatch)")
                continue
            
            source_df = pd.DataFrame(source_table['data'], columns=source_table['columns'])
            dest_df = pd.DataFrame(dest_table['data'], columns=dest_table['columns'])
            
            # Handle empty dataframes
            if source_df.empty and dest_df.empty:
                continue
            
            # Remove rows with all NULL values
            source_df = source_df.dropna(how='all')
            dest_df = dest_df.dropna(how='all')

            source_df = source_df.reindex(sorted(source_df.columns), axis=1)
            dest_df = dest_df.reindex(sorted(dest_df.columns), axis=1)

            # Reset index and sort
            source_df.reset_index(drop=True, inplace=True)
            dest_df.reset_index(drop=True, inplace=True)
            source_df = source_df.sort_values(by=source_df.columns.tolist()).reset_index(drop=True)
            dest_df = dest_df.sort_values(by=dest_df.columns.tolist()).reset_index(drop=True)

            if not source_df.equals(dest_df):
                mismatched_tables.append(f"{table_name} (data mismatch)")
        
        if mismatched_tables:
            return False, f"Mismatched tables: {', '.join(mismatched_tables)}", source_content, dest_content
        else:
            return True, "All tables in source and destination databases are identical", source_content, dest_content
    except Exception as e:
        return False, f"Error comparing databases: {str(e)}", None, None


if __name__ == '__main__':
    app.run(debug=True)
