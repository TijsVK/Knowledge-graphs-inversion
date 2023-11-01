import morph_kgc 
from morph_kgc.mapping.mapping_parser import retrieve_mappings
from morph_kgc.args_parser import load_config_from_argument
import pathlib
import json
import pandas as pd
import time
import warnings
import os
import sys
import pyrdf4j.rdf4j
import pyrdf4j.errors
from SPARQLWrapper import SPARQLWrapper, JSON
import re

rdf4jconnector = pyrdf4j.rdf4j.RDF4J(rdf4j_base="http://localhost:7200/")

warnings.simplefilter(action='ignore', category=FutureWarning)

sys.stdout = open("outbasic-inversion.txt", "w")

metadatapath = pathlib.Path("Implementation/rml-test-cases/metadata.csv")
assert metadatapath.exists(), "File \"Implementation/rml-test-cases/metadata.csv\" does not exist"
testcasesbasepath = pathlib.Path("Implementation/rml-test-cases/test-cases")
assert testcasesbasepath.exists(), "Directory \"Implementation/rml-test-cases/test-cases\" does not exist"
mappingfilename = "mapping.ttl"
outputfilename = "output.nq"

get_all_subjects_query = """
SELECT DISTINCT ?s
WHERE {
    ?s ?p ?o
}
"""

with open(metadatapath, "r") as file:
    testsDf:pd.DataFrame = pd.read_csv(file)

table_tests = testsDf[testsDf["data format"] == "CSV"]
table_tests_with_output = table_tests[table_tests["error expected?"] == False]

os.chdir(testcasesbasepath)
for _, row in table_tests_with_output.iterrows():
    os.chdir(row["RML id"])
    print(row["RML id"], row["title"], row["purpose"], row["error expected?"])
    config = f"""
            [CONFIGURATION]
            # INPUT
            na_values=,#N/A,N/A,#N/A N/A,n/a,NA,<NA>,#NA,NULL,null,NaN,nan,None

            # OUTPUT
            output_file=knowledge-graph.nt
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
            mappings: {mappingfilename}
         """
    
    # print(loaded_config.__dict__)
    try:
        rdf4jconnector.create_repository(row["better RML id"], accept_existing=True) # normal RML id
        loaded_config = load_config_from_argument(config)
        rules_df, fnml_df = retrieve_mappings(loaded_config)
        with open(mappingfilename, "r") as file:
            print(file.read())
        rule_subjects_regex_list = []
        for _, r in rules_df.iterrows():
            base = r["subject_map_value"]
            base = base.replace('/', '\/').replace('.', '\.')
            regexed = re.sub('{[^{]*}', '([^\/]*)', base) + '$'
            print(regexed)
            for key, value in r.items():
                print(f"{key}: {value}")
            print()
        with open(outputfilename, "r") as file:
            fileContents = file.read()
            print(fileContents)
            rdf4jconnector.add_data_to_repo(row["better RML id"], fileContents, "text/x-nquads")
        sparql = SPARQLWrapper(f"http://DESKTOP-IV4QGIH:7200/repositories/{row['better RML id']}")
        sparql.setReturnFormat(JSON)
        sparql.setQuery(get_all_subjects_query)
        subjects_dict = sparql.query().convert()["results"]["bindings"]
        subjects = [x["s"]["value"] for x in subjects_dict]
        
        print(json.dumps(subjects, indent=4))
    except pyrdf4j.errors.DataBaseNotReachable as e:
        print("You need to start a triplestore first.")
        quit()
    except Exception as e:
        print("Even though we filter the tests with error expected? == False, some tests still expectedly fail somehow.")
        print(type(e))
        print(e)
    rdf4jconnector.drop_repository(row["better RML id"], accept_not_exist=True)
    print('\n' * 5)
    os.chdir("..")
