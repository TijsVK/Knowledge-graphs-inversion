from flask import Flask, render_template, request, jsonify
from configparser import ConfigParser
import os
import csv
from r2rml_test_cases.test import test_one, database_up, database_down, generate_results, merge_results

app = Flask(__name__)

# Load configuration
config = ConfigParser()
config.read('config.ini')

TEST_CASES_DIR = os.path.join(os.path.dirname(__file__), 'r2rml_test_cases')

def get_mapping_filename(test_id):
    # Extract the letter from the test ID (e.g., 'a' from 'R2RMLTC0002a')
    letter = test_id[-1].lower()
    return f'r2rml{letter}.ttl'

@app.route('/')
def index():
    # Get list of available tests
    tests = [f for f in os.listdir(TEST_CASES_DIR) if os.path.isdir(os.path.join(TEST_CASES_DIR, f)) and f.startswith('R2RMLTC')]
    return render_template('index.jinja', tests=tests)

@app.route('/run_test', methods=['POST'])
def run_test():
    test_id = request.form['test_id']
    database_system = request.form['database_system']
    
    # Update config
    config['properties']['database_system'] = database_system
    config['properties']['tests'] = test_id
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    # Change to the test case directory
    test_dir = os.path.join(TEST_CASES_DIR, test_id)
    os.chdir(test_dir)
    
    # Run test
    try:
        database_up()
        test_one(test_id)
        generate_results()
        database_down()
        merge_results()
        
        # Read results
        with open(f'results-{database_system}.csv', 'r') as f:
            reader = csv.reader(f)
            results = list(reader)
        
        # Change back to the original directory
        os.chdir(os.path.dirname(__file__))
        
        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        # Change back to the original directory
        os.chdir(os.path.dirname(__file__))
        return jsonify({'status': 'error', 'message': str(e)})

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