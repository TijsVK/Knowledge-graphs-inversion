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
import io
import urllib.parse

FAST_TEST = False


def compare_dfs(df1:pd.DataFrame, df2:pd.DataFrame):
    # sort both dataframes by columns
    df1.sort_index(axis=1, inplace=True)
    df2.sort_index(axis=1, inplace=True)
    if df1.shape[0] != df2.shape[0]:
        raise IndexError(f"DataFrames have different row counts: {df1.shape[0]} (should be {df2.shape[0]})")
    elif df1.shape[1] != df2.shape[1]:
        raise IndexError(f"DataFrames have different column counts: {df1.shape[1]} vs {df2.shape[1]}, {df1.columns} != {df2.columns}")
    # for each row in df1, check if it exists in df2
    for row in df1.itertuples():
        if row not in df2.itertuples():
            raise ValueError(f"Row {row} from df1 not found in df2")

get_all_subjects_query = """
    SELECT DISTINCT ?s
    WHERE {
        ?s ?p ?o
    }
"""

def get_all_data_for_subject_query(subject):
    #subject = urllib.parse.quote(subject)
    return f"""
SELECT ?p ?o
WHERE {{
    <{subject}> ?p ?o
}}
"""

def main():
    rdf4jconnector = pyrdf4j.rdf4j.RDF4J(rdf4j_base="http://localhost:7200/")

    warnings.simplefilter(action='ignore', category=FutureWarning)

    if(not FAST_TEST):
        sys.stdout = open("basic-inversion-output.txt", "w")

    thisFilePath = pathlib.Path(__file__).resolve()
    ImplementationDir = thisFilePath.parent
    metadatapath = ImplementationDir / "rml-test-cases" / "metadata.csv"
    assert metadatapath.exists(), f"File \"Implementation/rml-test-cases/metadata.csv\" does not exist  (tried {metadatapath})"
    testcasesbasepath = ImplementationDir / "rml-test-cases" / "test-cases"
    assert testcasesbasepath.exists(), f"Directory \"Implementation/rml-test-cases/test-cases\" does not exist (tried {testcasesbasepath})"
    mappingfilename = "mapping.ttl"
    outputfilename = "output.nq"

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
            with open(outputfilename, "r") as file:
                fileContents = file.read()
                print(fileContents)
                rdf4jconnector.add_data_to_repo(row["better RML id"], fileContents, "text/x-nquads")
            sparql = SPARQLWrapper(f"http://DESKTOP-IV4QGIH:7200/repositories/{row['better RML id']}")
            sparql.setReturnFormat(JSON)
            loaded_config = load_config_from_argument(config)
            rules_df:pd.DataFrame
            rules_df, fnml_df = retrieve_mappings(loaded_config)
            with open(mappingfilename, "r") as file:
                print(file.read())
                
            # add columns to end of dataframe, probably faster than inserting them in the middle but worse for overview
            # rules_df["subject_references"] = [[] for _ in range(rules_df.shape[0])]
            # rules_df["subject_reference_count"] = 0
            # rules_df["predicate_references"] = [[] for _ in range(rules_df.shape[0])]
            # rules_df["predicate_reference_count"] = 0
            # rules_df["object_references"] = [[] for _ in range(rules_df.shape[0])]
            # rules_df["object_reference_count"] = 0
            
            # add columns to dataframe at specific index, probably slower than adding them to the end but better for overview when printing rows
            rules_df.insert(rules_df.columns.get_loc('subject_map_value') + 1, "subject_references",  [[] for _ in range(rules_df.shape[0])])
            rules_df.insert(rules_df.columns.get_loc('subject_references') + 1, "subject_reference_count", 0)
            rules_df.insert(rules_df.columns.get_loc('predicate_map_value') + 1, "predicate_references", [[] for _ in range(rules_df.shape[0])])
            rules_df.insert(rules_df.columns.get_loc('predicate_references') + 1, "predicate_reference_count", 0)
            rules_df.insert(rules_df.columns.get_loc('object_map_value') + 1, "object_references", [[] for _ in range(rules_df.shape[0])])
            rules_df.insert(rules_df.columns.get_loc('object_references') + 1, "object_reference_count", 0)
            
            for index in rules_df.index:
                if rules_df.at[index, "subject_map_type"] == "http://w3id.org/rml/constant":
                    rules_df.at[index, "subject_references"] = []
                    rules_df.at[index, "subject_reference_count"] = 0
                    
                elif rules_df.at[index, "subject_map_type"] == "http://w3id.org/rml/reference":
                    rules_df.at[index, "subject_references"] = [rules_df.at[index, "subject_map_value"]]
                    rules_df.at[index, "subject_reference_count"] = 1
                    
                elif rules_df.at[index, "subject_map_type"] == "http://w3id.org/rml/template":
                    referencesList = re.findall("{([^{]*)}", rules_df.at[index, "subject_map_value"])
                    rules_df.at[index, "subject_references"] = referencesList
                    rules_df.at[index, "subject_reference_count"] = len(referencesList)
                    
                if rules_df.at[index, "predicate_map_type"] == "http://w3id.org/rml/constant":
                    rules_df.at[index, "predicate_references"] = []
                    rules_df.at[index, "predicate_reference_count"] = 0
                    
                elif rules_df.at[index, "predicate_map_type"] == "http://w3id.org/rml/reference":
                    rules_df.at[index, "predicate_references"] = [rules_df.at[index, "predicate_map_value"]]
                    rules_df.at[index, "predicate_reference_count"] = 1
                    
                elif rules_df.at[index, "predicate_map_type"] == "http://w3id.org/rml/template":
                    referencesList = re.findall("{([^{]*)}", rules_df.at[index, "predicate_map_value"])
                    rules_df.at[index, "predicate_references"] = referencesList
                    rules_df.at[index, "predicate_reference_count"] = len(referencesList)
                  
                    
                if rules_df.at[index, "object_map_type"] == "http://w3id.org/rml/constant":
                    rules_df.at[index, "object_references"] = []
                    rules_df.at[index, "object_reference_count"] = 0
                    
                elif rules_df.at[index, "object_map_type"] == "http://w3id.org/rml/reference":
                    rules_df.at[index, "object_references"] = [rules_df.at[index, "object_map_value"]]
                    rules_df.at[index, "object_reference_count"] = 1
                    
                elif rules_df.at[index, "object_map_type"] == "http://w3id.org/rml/template":
                    referencesList = re.findall("{([^{]*)}", rules_df.at[index, "object_map_value"])
                    rules_df.at[index, "object_references"] = referencesList
                    rules_df.at[index, "object_reference_count"] = len(referencesList)
                    
                elif rules_df.at[index, "object_map_type"] == "http://w3id.org/rml/parentTriplesMap":
                    rules_df.at[index, "object_references"] = [list(json.loads(rules_df.at[index, "object_join_conditions"].replace("'", '"')).values())[0]['child_value']]
                    rules_df.at[index, "object_reference_count"] = 1
            
            for source, rules in rules_df.groupby("logical_source_value"):
                with open(source, "r") as file:
                    originalSource = file.read()
                print(f"{source}: {len(rules)} mapping(s)")
                sourceReferences = set()
                for _, rule in rules.iterrows():
                    for reference in rule["subject_references"]:
                        sourceReferences.add(reference)
                    for reference in rule["predicate_references"]:
                        sourceReferences.add(reference)
                    for reference in rule["object_references"]:
                        sourceReferences.add(reference)
                print(f"{source}: {len(sourceReferences)} reference(s): {sourceReferences}")
                
                sparql.setQuery(get_all_subjects_query)
                subjects_dict = sparql.query().convert()["results"]["bindings"]
                subjects = [x["s"]["value"] for x in subjects_dict]
                print(json.dumps(subjects, indent=4))
                
                originalSourceDf = pd.read_csv(io.StringIO(originalSource))
                originalSourceDf.sort_index(axis=1, inplace=True)
                outputSourceDf = pd.DataFrame(columns=list(sourceReferences))
                outputSourceDf.sort_index(axis=1, inplace=True)
                
                for subjectTemplate, subjectRules in rules.groupby("subject_map_value"):
                    subjectType = subjectRules.at[subjectRules.index[0], "subject_map_type"]
                    subjectTermType = subjectRules.at[subjectRules.index[0], "subject_termtype"]
                    if subjectTermType == "http://w3id.org/rml/BlankNode":
                        print(f"BlankNode subjects are not supported yet.")
                        continue
                    if subjectType == "http://w3id.org/rml/template":
                        matchingSubjects = []
                        object_map_value = rules_df.at[index, "subject_map_value"]
                        template = re.sub('{[^{]*}', '([^\/]*)', object_map_value) + '$'
                        for s in subjects:
                            if re.match(template, s):
                                matchingSubjects.append(s)
                        print(f"{subjectTemplate}[{template}]: {len(matchingSubjects)} matching subject(s): {json.dumps(matchingSubjects, indent=4)}")
                    elif subjectType == "http://w3id.org/rml/reference":
                        print(f"Reference based subjects are not supported yet.")
                        continue
                    elif subjectType == "http://w3id.org/rml/constant":
                        print(f"Constant subjects are not supported yet.")
                        continue
                    
                    for subject in matchingSubjects:
                        # print(get_all_data_for_subject_query(subject))
                        sparql.setQuery(get_all_data_for_subject_query(subject))
                        data_dict = sparql.query().convert()["results"]["bindings"]
                        data = {x["p"]["value"]: x["o"]["value"] for x in data_dict}
                        print(f"{subjectTemplate}: {len(data)} data piece(s): {json.dumps(data, indent=4)}")
                    
                    print(f"{subjectTemplate}: {len(subjectRules)} mapping(s)")
                    print(f"Rules:")
                    for _, rule in subjectRules.iterrows():
                        for key, value in rule.items():
                            print(f"\t{key}: {value}")
                        print()
                    
                    

                print(outputSourceDf.head())
                print(f"Source: {source}")
                print(originalSourceDf.to_csv(index=False, lineterminator='\n'))
                print(f"Output: out_{source}")
                print(outputSourceDf.to_csv(index=False, lineterminator='\n'))
                try:
                    compare_dfs(outputSourceDf, originalSourceDf)
                    print("Dataframes are equal WOOHOO")
                except IndexError as e:
                    print(e)
                except ValueError as e:
                    print(e)
                    
                
                
            
            
        except pyrdf4j.errors.DataBaseNotReachable as e:
            print("You need to start a triplestore first.")
            quit()
        except ValueError as e:
            print("Even though we filter the tests with error expected? == False, some tests still expectedly fail. Morhp-KGC isnt able to handle all the tests either, so those error too sometimes.")
            print(type(e))
            print(e)
            if str(e) != "Found an invalid graph termtype. Found values {'http://w3id.org/rml/BlankNode'}. Graph maps must be http://w3id.org/rml/IRI.":
                # morph-kgc uses build-in python errors, but catching all ValueErrors is not a good idea as it also catches errors that are not related to morph-kgc
                raise
        rdf4jconnector.drop_repository(row["better RML id"], accept_not_exist=True)
        print('\n' * 5)
        os.chdir("..")
        if(FAST_TEST):
            break


if __name__ == "__main__":
    main()