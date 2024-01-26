from flask import Flask, render_template, request, send_from_directory
from turbo_flask import Turbo

import pathlib

import sys, os

import pretty_yarrrml2rml
import yaml

app = Flask(__name__)
turbo = Turbo(app)

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

@app.route('/')
def index():
    return render_template('demo.html', sources=['Glassdoor'])

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
        rml_content = pretty_yarrrml2rml.translate(yaml.safe_load(yarrrml))
    push_generated_mappings(rml_content)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/load_preset', methods=['POST'])
def load_preset():
    preset = request.form['preset']
    if preset == 'None':
        push_yarrml("")
        push_generated_mappings("")
        push_source_name("")
        push_source_content("")
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

if __name__ == '__main__':
    app.run(debug=True)