import pyrdf4j.rdf4j as rdf4j
import pathlib
import json
import pandas as pd
import time
import os


rdf4jconnector = rdf4j.RDF4J(rdf4j_base="http://localhost:7200/")

metadatapath = pathlib.Path("Implementation/rml-test-cases/metadata.csv")
testcasesbasepath = pathlib.Path("Implementation/rml-test-cases/test-cases")
mappingfilename = "mapping.ttl"
outputfilename = "output.nq"

with open(metadatapath, "r") as file:
    testsDf:pd.DataFrame = pd.read_csv(file)

table_tests = testsDf[testsDf["data format"] == "CSV"]
table_tests_with_output = table_tests[table_tests["error expected?"] == False]

os.chdir(testcasesbasepath)
for _, row in table_tests_with_output.iterrows():
    os.chdir(row["RML id"])
    try:
        rdf4jconnector.create_repository(row["better RML id"], accept_existing=True) # normal RML id contains a dash, which is not allowed in the repository id (as it is seen an some kind of config)
        print(row["RML id"], row["better RML id"], row["title"], row["purpose"])
        with open(outputfilename, "r") as file:
            fileContents = file.read()
            rdf4jconnector.add_data_to_repo(row["better RML id"], fileContents, "text/x-nquads")
        rdf4jconnector.drop_repository(row["better RML id"])
    except Exception as e:
        print("Even though we filter the tests with error expected? == False, some tests still expectedly fail somehow.")
        print(type(e), e)
    print('\n' * 2)
    os.chdir("..")

