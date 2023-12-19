import time

import sys
sys.path.append("F:\\Github_repos\\Inversion_KG_to_raw_data\\Implementation\\morph-kgc\\src")

from morph_kgc.mapping.mapping_parser import retrieve_mappings
from morph_kgc.args_parser import load_config_from_argument
import pathlib
import json
import pandas as pd
import warnings
import os
import pyrdf4j.rdf4j
import pyrdf4j.errors
import rdflib
from SPARQLWrapper import SPARQLWrapper, JSON, CSV
import re
import cProfile
import validators
from urllib.parse import urlparse
from io import StringIO

# region Setup
pyrdf4j.repo_types.REPO_TYPES = pyrdf4j.repo_types.REPO_TYPES + ["graphdb"]  # add graphdb to the list of repo types

MORPH_CONFIG = f"""
    [CONFIGURATION]
    # INPUT
    na_values=,#N/A,N/A,#N/A N/A,n/a,NA,<NA>,#NA,NULL,null,NaN,nan,None

    # OUTPUT
    output_file=output.nq
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
    mappings: mapping.ttl
"""

REPO_ID = "inversion"
TRIPLESTORE_URL = f"http://localhost:7200/repositories/{REPO_ID}"

TEST_CASES_PATH = "F:\\Github_repos\\Inversion_KG_to_raw_data\\Implementation\\rml-test-cases\\test-cases"

# endregion

def compare_dfs(df1: pd.DataFrame, df2: pd.DataFrame):
    # sort both dataframes by columns
    df1.sort_index(axis=1, inplace=True)
    df1.sort_values(by=list(df1.columns), inplace=True)
    df2.sort_index(axis=1, inplace=True)
    df2.sort_values(by=list(df2.columns), inplace=True)
    if df1.shape[0] != df2.shape[0]:
        raise IndexError(f"DataFrames have different row counts: {df1.shape[0]} (should be {df2.shape[0]})")
    elif df1.shape[1] != df2.shape[1]:
        raise IndexError(
            f"DataFrames have different column counts: {df1.shape[1]} vs {df2.shape[1]}, "
            f"{df1.columns} != {df2.columns}")
    # for each row in df1, check if it exists in df2
    for row in df1.itertuples():
        if row not in df2.itertuples():
            raise ValueError(f"Row {row} from df1 not found in df2")
    
    for row in df2.itertuples():
        if row not in df1.itertuples():
            raise ValueError(f"Row {row} from df2 not found in df1")
        
    return True


def uri_validator(x) -> bool:
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False
    
def create_sparql_endpoint(knowledge_graph_path:str) -> SPARQLWrapper:
    knowledge_graph = knowledge_graph_path.strip()
    if uri_validator(knowledge_graph):
        endpoint = SPARQLWrapper(knowledge_graph)
    else:
        rdf4jconnector = pyrdf4j.rdf4j.RDF4J(rdf4j_base="http://localhost:7200/")
        rdf4jconnector.empty_repository(REPO_ID)
        rdf4jconnector.create_repository(REPO_ID, accept_existing=True, repo_type='graphdb')
        data = open(knowledge_graph_path, "r").read()
        rdf4jconnector.add_data_to_repo(REPO_ID, data, "text/x-nquads")
        
        endpoint = SPARQLWrapper(TRIPLESTORE_URL)
    endpoint.setReturnFormat(CSV)
    return endpoint

def insert_columns(df: pd.DataFrame, pure=False) -> pd.DataFrame:
    if pure:
        df = df.copy(deep=True)  # do not modify original dataframe (pure function)
    # add columns to end of dataframe, probably faster than inserting them in the middle but worse for overview
    # rules_df["subject_references"] = [[] for _ in range(rules_df.shape[0])]
    # rules_df["subject_reference_count"] = 0
    # rules_df["predicate_references"] = [[] for _ in range(rules_df.shape[0])]
    # rules_df["predicate_reference_count"] = 0
    # rules_df["object_references"] = [[] for _ in range(rules_df.shape[0])]
    # rules_df["object_reference_count"] = 0

    # add columns to dataframe at specific index
    # probably slower than adding them to the end but better for overview when printing rows
    df.insert(df.columns.get_loc('subject_map_value') + 1, "subject_references", [[] for _ in range(df.shape[0])])
    df.insert(df.columns.get_loc('subject_map_value') + 1, "subject_references_template", None)
    df.insert(df.columns.get_loc('subject_references') + 1, "subject_reference_count", 0)
    df.insert(df.columns.get_loc('predicate_map_value') + 1, "predicate_references", [[] for _ in range(df.shape[0])])
    df.insert(df.columns.get_loc('predicate_map_value') + 1, "predicate_references_template", None)
    df.insert(df.columns.get_loc('predicate_references') + 1, "predicate_reference_count", 0)
    df.insert(df.columns.get_loc('object_map_value') + 1, "object_references", [[] for _ in range(df.shape[0])])
    df.insert(df.columns.get_loc('object_map_value') + 1, "object_references_template", None)
    df.insert(df.columns.get_loc('object_references') + 1, "object_reference_count", 0)

    for index in df.index:
        match df.at[index, "subject_map_type"]:
            case "http://w3id.org/rml/constant":
                df.at[index, "subject_references"] = []
                df.at[index, "subject_reference_count"] = 0

            case "http://w3id.org/rml/reference":
                df.at[index, "subject_references"] = [df.at[index, "subject_map_value"]]
                df.at[index, "subject_reference_count"] = 1

            case "http://w3id.org/rml/template":
                references_list = re.findall("{([^{]*)}", df.at[index, "subject_map_value"])
                df.at[index, "subject_references"] = references_list
                df.at[index, "subject_reference_count"] = len(references_list)
                df.at[index, "subject_references_template"] = re.sub('{[A-z0-9^{}]*}', '([^\/]*)', df.at[index, "subject_map_value"]) + '$'
                
        match df.at[index, "predicate_map_type"]:
            case "http://w3id.org/rml/constant":
                df.at[index, "predicate_references"] = []
                df.at[index, "predicate_reference_count"] = 0

            case "http://w3id.org/rml/reference":
                df.at[index, "predicate_references"] = [df.at[index, "predicate_map_value"]]
                df.at[index, "predicate_reference_count"] = 1

            case "http://w3id.org/rml/template":
                references_list = re.findall("{([^{]*)}", df.at[index, "predicate_map_value"])
                df.at[index, "predicate_references"] = references_list
                df.at[index, "predicate_reference_count"] = len(references_list)
                df.at[index, "predicate_references_template"] = re.sub('{[A-z0-9^{}]*}', '([^\/]*)', df.at[index, "predicate_map_value"]) + '$'

        match df.at[index, "object_map_type"]:
            case "http://w3id.org/rml/constant":
                df.at[index, "object_references"] = []
                df.at[index, "object_reference_count"] = 0

            case "http://w3id.org/rml/reference":
                df.at[index, "object_references"] = [df.at[index, "object_map_value"]]
                df.at[index, "object_reference_count"] = 1

            case "http://w3id.org/rml/template":
                references_list = re.findall("{([^{]*)}", df.at[index, "object_map_value"])
                df.at[index, "object_references"] = references_list
                df.at[index, "object_reference_count"] = len(references_list)
                df.at[index, "object_references_template"] = re.sub('{[A-z0-9^{}]*}', '([^\/]*)', df.at[index, "object_map_value"]) + '$'

            case "http://w3id.org/rml/parentTriplesMap":
                df.at[index, "object_references"] = [
                    list(json.loads(df.at[index, "object_join_conditions"].replace("'", '"')).values())[0]['child_value']]
                df.at[index, "object_reference_count"] = 1
                
    return df

def get_references(rules:pd.DataFrame) -> set:
    references = set()
    for _, rule in rules.iterrows():
        for reference in rule["subject_references"]:
            references.add(reference)
        for reference in rule["predicate_references"]:
            references.add(reference)
        for reference in rule["object_references"]:
            references.add(reference)
    return references

def generate_object_fragment(rule:pd.Series, subject_index:int) -> str:
    temp_index_counter = 0
    subject = f"?s{subject_index}"
    predicate = f"<{rule['predicate_map_value']}>" # technically this can be not constant, but we ignore that for now as it is very rare
    match rule["object_map_type"]:
        case "http://w3id.org/rml/constant":
            if rule["object_termtype"] == "http://w3id.org/rml/IRI":
                return f"{subject} {predicate} <{rule['object_map_value']}> ."
            elif rule["object_termtype"] == "http://w3id.org/rml/Literal":
                return f"{subject} {predicate} {rule['object_map_value']} ."
            elif rule["object_termtype"] == "http://w3id.org/rml/BlankNode":
                raise NotImplementedError("BlankNodes not implemented yet")
                # TODO: Think about this
        
        case "http://w3id.org/rml/reference":
            return f"OPTIONAL{{{subject} {predicate} ?{rule['object_map_value']}}} ."
        
        case "http://w3id.org/rml/template":
            lines = []
            full_template_value = f"?temp{temp_index_counter}"
            temp_index_counter += 1
            lines.append(f"{subject} {predicate} {full_template_value} .")
            lines.append(f"FILTER(REGEX(STR({full_template_value}), '{rule['object_references_template']}'))")
            # concat_contents = rule["object_map_value"].replace("{", "\", ?").replace("}", ", \"")
            # lines.append(f"FILTER({full_template_value} = CONCAT(\"{concat_contents}\"))")
            last_cut = rule['object_map_value']
            last_variable = full_template_value
            pre_string = last_cut.split('{', 1)[0]
            for reference in rule["object_references"]:
                last_cut = last_cut.split('}', 1)[1]
                if last_cut == "":
                    lines.append(f"BIND(STRAFTER(STR({last_variable}), '{pre_string}') AS ?{reference})")
                else:
                    lines.append(f"BIND(STRAFTER(STR({last_variable}), '{pre_string}') AS ?temp{temp_index_counter})")
                    pre_string = last_cut.split('{', 1)[0]
                    lines.append(f"BIND(STRBEFORE(STR({last_variable}), '{pre_string}') AS ?{reference})")
                    last_variable = f"?temp{temp_index_counter}"
                    temp_index_counter += 1

            return "\n".join(lines)
               
def generate_subject_fragment(rule:pd.Series, subject_index:int) -> str:
    if rule["subject_termtype"] == "http://w3id.org/rml/IRI":
        if rule["subject_map_type"] == "http://w3id.org/rml/template":
            temp_index_counter = 0
            lines = []
            full_template_value = f"?s{subject_index}"
            temp_index_counter += 1
            lines.append(f"FILTER(REGEX(STR({full_template_value}), '{rule['subject_references_template']}'))")
            last_cut = rule['subject_map_value']
            last_variable = full_template_value
            pre_string = last_cut.split('{', 1)[0]
            for reference in rule["subject_references"]:
                last_cut = last_cut.split('}', 1)[1]
                lines.append(f"BIND(STRAFTER(STR({last_variable}), '{pre_string}') AS ?s{subject_index}_temp{temp_index_counter})")
                last_variable = f"?s{subject_index}_temp{temp_index_counter}"
                temp_index_counter += 1
                if last_cut == "":
                    lines.append(f"FILTER({last_variable} = ?{reference})")
                    lines.append(f"OPTIONAL{{BIND({last_variable} as ?{reference})}}")
                else:
                    pre_string = last_cut.split('{', 1)[0]
                    lines.append(f"BIND(STRBEFORE(STR({last_variable}), '{pre_string}') AS ?s{subject_index}_temp{temp_index_counter})")
                    lines.append(f"FILTER(?s{subject_index}_temp{temp_index_counter} = ?{reference})")
                    lines.append(f"OPTIONAL{{BIND(?s{subject_index}_temp{temp_index_counter} as ?{reference})}}")
                    temp_index_counter += 1

            return "\n".join(lines)
    return None

def generate_query(iterator_rules:pd.DataFrame) -> str:
    references = get_references(iterator_rules)
    if len(references) == 0:
        return None
    lines = []
    grouped_by_subject = iterator_rules.groupby("subject_map_value")
    for i, (subject, subject_rules) in enumerate(grouped_by_subject):
        for _, rule in subject_rules.iterrows():
            fragment = generate_object_fragment(rule, i)
            if fragment is not None:
                lines.append(fragment)
        else:
            fragment = generate_subject_fragment(rule, i)
            if fragment is not None:
                lines.append(fragment)
    lines = ["\t" + line.replace("\n", "\n\t") for line in lines]
    lines.append("}")
    lines.insert(0, f"SELECT {' '.join([f'?{reference}' for reference in references])} WHERE {{")
    query = "\n".join(lines)
    if query is None:
        return query
    else:
        return query.replace("\\", "\\\\")

# SELECT ?ID ?FirstName ?LastName WHERE {
#  ?s1 a <http://example.org/Person> .
# }


def invert_source(source_rules:pd.DataFrame, sparql:SPARQLWrapper):
    for iterator, iterator_rules in source_rules.groupby('iterator', dropna=False):
        query = generate_query(iterator_rules)
        print(query)
        if query is None:
            print("No query generated (no references found)")
            iterator_result = None
        else:
            sparql.setQuery(query)
            iterator_result = sparql.query().convert().decode('utf-8')
    # generate template
    # fill template with values
    # we dont do any of this yet (PoC)
    return iterator_result
    

    
    

def inversion(config_file: str|pathlib.Path):
    config = load_config_from_argument(config_file)
    mappings, _ = retrieve_mappings(config)
    knowledge_graph_file = config.get_output_file()
    sparql = create_sparql_endpoint(knowledge_graph_file)
    insert_columns(mappings)
    for source, source_rules in mappings.groupby("logical_source_value"):
        generated_source = invert_source(source_rules, sparql)


        # after this should really be in its own function
        if generated_source is None:
            generated_df = pd.DataFrame()
        else:
            generated_df = pd.read_csv(StringIO(generated_source))     # this is only for csv files
        with open(source, "r") as file:
            expected_source = pd.read_csv(file)
        try:
            print("-" * os.get_terminal_size().columns)
            print(generated_df)
            print("-" * os.get_terminal_size().columns)
            print(expected_source)
            print("-" * os.get_terminal_size().columns)
            compare_dfs(generated_df, expected_source)
        except ValueError as e:
            print(e)
            print("Test failed")
        except IndexError as e:
            print(e)
            print("Test failed")
        else:
            print("Dataframes are equal")
            print("Test passed")
        print("+" * os.get_terminal_size().columns)
def test_rml_test_cases():
    os.chdir(TEST_CASES_PATH)
    this_file_path = pathlib.Path(__file__).resolve()
    implementation_dir = this_file_path.parent
    metadata_path = implementation_dir / "rml-test-cases" / "metadata.csv"
    testcases_path = implementation_dir / "rml-test-cases" / "test-cases"
    
    with open(metadata_path, "r") as file:
        tests_df: pd.DataFrame = pd.read_csv(file)
    
    table_tests = tests_df[tests_df["data format"] == "CSV"]
    table_tests_with_output = table_tests[table_tests["error expected?"] == False]
    
    os.chdir(testcases_path)
    for _, row in table_tests_with_output.iterrows():
        print(f'Running test {row["RML id"]}, ({row["better RML id"]})')
        os.chdir(testcases_path / row["RML id"])
        try:
            inversion(MORPH_CONFIG)
        except Exception as e:
            print(e)
            print("Test failed")
        
        
    

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    test_rml_test_cases()
    

if __name__ == '__main__':
    main()