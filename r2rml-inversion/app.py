from flask import Flask, render_template, request, jsonify
from configparser import ConfigParser
import os
from rdflib import ConjunctiveGraph, Namespace, Literal
import traceback
from r2rml_test_cases.test import test_one, generate_results, database_load
import docker
from docker.errors import ImageNotFound, APIError

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

class DatabaseManager:
    def __init__(self, database_system):
        self.database_system = database_system
        self.client = docker.from_env()
        self.container = None

    def start_container(self):
        if self.container is None or not self.container_is_running():
            try:
                image = 'postgres:13' if self.database_system == 'postgresql' else 'mysql:8'
                self.container = self.client.containers.run(
                    image,
                    detach=True,
                    remove=True,
                    environment={
                        'POSTGRES_PASSWORD': 'r2rml',
                        'POSTGRES_USER': 'r2rml',
                        'POSTGRES_DB': 'r2rml'
                    } if self.database_system == 'postgresql' else {
                        'MYSQL_ROOT_PASSWORD': 'r2rml',
                        'MYSQL_USER': 'r2rml',
                        'MYSQL_PASSWORD': 'r2rml',
                        'MYSQL_DATABASE': 'r2rml'
                    },
                    ports={'5432/tcp': 5432} if self.database_system == 'postgresql' else {'3306/tcp': 3306}
                )
                print(f"Container started: {self.container.id}")
            except ImageNotFound:
                print(f"Error: Docker image not found. Please ensure you have the {image} image pulled.")
                raise
            except APIError as e:
                print(f"Error starting container: {e}")
                raise

    def container_is_running(self):
        if self.container:
            self.container.reload()
            return self.container.status == 'running'
        return False

    def stop_container(self):
        if self.container:
            try:
                self.container.stop()
                print(f"Container stopped: {self.container.id}")
                self.container = None
            except APIError as e:
                print(f"Error stopping container: {e}")

    def reset_database(self):
        # Implement database reset logic here
        pass

def get_mapping_filename(test_id):
    letter = test_id[-1].lower()
    return f'r2rml{letter}.ttl'

@app.route('/')
def index():
    tests = [f for f in os.listdir(TEST_CASES_DIR) if os.path.isdir(os.path.join(TEST_CASES_DIR, f)) and f.startswith('R2RMLTC')]
    return render_template('index.jinja', tests=tests)

# In your Flask route:
@app.route('/run_test', methods=['POST'])
def run_test():
    test_id = request.form['test_id']
    database_system = request.form['database_system']
    
    db_manager = DatabaseManager(database_system)
    
    test_dir = os.path.join(TEST_CASES_DIR)
    os.chdir(test_dir)

    try:
        db_manager.start_container()
        
        # Wait for the database to be ready
        import time
        time.sleep(10)  # Adjust this value as needed
        
        # Load the specific test database
        test_uri = manifest_graph.value(subject=None, predicate=DCELEMENTS.identifier, object=Literal(test_id))
        database_uri = manifest_graph.value(subject=test_uri, predicate=RDB2RDFTEST.database, object=None)
        database = manifest_graph.value(subject=database_uri, predicate=RDB2RDFTEST.sqlScriptFile, object=None)
        database_load(database, database_system)
        
        results = test_one(test_id, database_system, config, manifest_graph)
        generate_results(database_system, config, results)
        
        # Reset database for next test
        db_manager.reset_database()
        
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
    finally:
        os.chdir(os.path.dirname(__file__))
        db_manager.stop_container()

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