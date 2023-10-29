import morph_kgc 
from morph_kgc.mapping.mapping_parser import retrieve_mappings
from morph_kgc.args_parser import load_config_from_argument
import pathlib
import json
import pandas as pd
import time
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)


metadatapath = pathlib.Path("Implementation/rml-test-cases/metadata.csv")
assert metadatapath.exists(), "File \"Implementation/rml-test-cases/metadata.csv\" does not exist"
testcasesbasepath = pathlib.Path("Implementation/rml-test-cases/test-cases")
assert testcasesbasepath.exists(), "Directory \"Implementation/rml-test-cases/test-cases\" does not exist"
mappingfilename = "mapping.ttl"

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
        loaded_config = load_config_from_argument(config)
        rml_df, fnml_df = retrieve_mappings(loaded_config)
        with open(mappingfilename, "r") as file:
            print(file.read())
        for _, row in rml_df.iterrows():
            for key, value in row.items():
                print(f"{key}: {value}")
        graph = morph_kgc.materialize(config)
        print(graph.serialize(format="ntriples"))
    except Exception as e:
        print("Even though we filter the tests with error expected? == False, some tests still expectedly fail somehow.")
        print(e)
    print('\n' * 5)
    os.chdir("..")