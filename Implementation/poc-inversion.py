from abc import ABC, abstractmethod
import time
from typing import Any
from xml.dom.minidom import Document

import morph_kgc.config
from morph_kgc.mapping.mapping_parser import retrieve_mappings
from morph_kgc.args_parser import load_config_from_argument
import pathlib
import json
import pandas as pd
import warnings
import os
import pyrdf4j.rdf4j
import pyrdf4j.errors
import pyrdf4j.repo_types
import rdflib
from SPARQLWrapper import SPARQLWrapper, CSV
import re
from urllib.parse import ParseResult, urlparse
from io import StringIO
import hashlib
import logging

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

REF_TEMPLATE_REGEX = '{([^{}]*)}'

# endregion

# region classes

class IdGenerator:
    counter = 0
    
    @classmethod
    def get_id(cls):
        cls.counter += 1
        return cls.counter

    @classmethod
    def reset(cls):
        cls.counter = 0


"""
properties:
    references: list[reference]
"""
class QueryFragment():
    def __init__(self):
        pass
    
    @property
    def references(self) -> set[str]:
        return set()

class ConstantQueryFragment(QueryFragment):
    def __init__(self, subject:str, predicate:str, object:str):
        super().__init__()

class Query:
    def __init__(self):
        self.fragments: list[QueryFragment] = []
        
    @property
    def references(self) -> set[str]:
        references = set()
        for fragment in self.fragments:
            references.update(fragment.references)
        return references
    
    def hashed_references(self) -> dict[str, str]:
        hashed_references = {}
        for reference in self.references:
            hashed_references[reference] = hashlib.md5(reference.encode('utf-8')).hexdigest()
        return hashed_references

class Endpoint(ABC):
    @abstractmethod
    def query(self, query: str):
        raise NotImplementedError
    
class RemoteEndpoint(Endpoint):
    def __init__(self, url:str):
        self._sparql = SPARQLWrapper(url)
        self._sparql.setReturnFormat(CSV)
    
    def query(self, query: str):
        self._sparql.setQuery(query)
        return self._sparql.query().convert().decode("utf-8")
    
    def __repr__(self):
        return f"RemoteSparqlEndpoint({self._sparql.endpoint})"

class LocalSparqlGraphStore(Endpoint):
    def __init__(self, url:str):
        data = open(url, "r").read()
        self._repoid = hashlib.md5(data.encode('utf-8')).hexdigest()
        rdf4jconnector = pyrdf4j.rdf4j.RDF4J(rdf4j_base="http://localhost:7200/")
        rdf4jconnector.empty_repository(self._repoid)
        rdf4jconnector.create_repository(self._repoid, accept_existing=True, repo_type='graphdb')
        rdf4jconnector.add_data_to_repo(self._repoid, data, "text/x-nquads")
        self._sparql = SPARQLWrapper(f"http://localhost:7200/repositories/{self._repoid}")
        self._sparql.setReturnFormat(CSV)
    
    def query(self, query: str) -> str:
        self._sparql.setQuery(query)
        query_result = self._sparql.query()
        converted:Any = query_result.convert()
        decoded = converted.decode("utf-8")
        return decoded
    
    def __del__(self):
        rdf4jconnector = pyrdf4j.rdf4j.RDF4J(rdf4j_base="http://localhost:7200/")
        rdf4jconnector.drop_repository(self._repoid)

    def __repr__(self):
        return f"LocalSparqlGraphStore({self._repoid})"

class Validator:
    @staticmethod
    def url(x) -> bool:
        try:
            result:ParseResult = urlparse(x)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def df_equals(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        # pure function
        df1 = df1.copy(deep=True)
        df2 = df2.copy(deep=True)
        # sort by columns and rows
        df1.sort_index(axis=1, inplace=True)
        df1.sort_values(by=list(df1.columns), inplace=True)
        df2.sort_index(axis=1, inplace=True)
        df2.sort_values(by=list(df2.columns), inplace=True)
        if df1.shape != df2.shape:
            return False
        # for each row in df1, check if it exists in df2
        for row in df1.itertuples():
            if row not in df2.itertuples():
                return False
        
        for row in df2.itertuples():
            if row not in df1.itertuples():
                return False
            
        return True

class EndpointFactory:
    @staticmethod
    def create(config:morph_kgc.config.Config):
        url = config.get_output_file()
        if Validator.url(url):
            return RemoteEndpoint(url)
        else:
            return LocalSparqlGraphStore(url)

# endregion

inversion_logger = logging.getLogger("inversion")

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
                references_list = re.findall(REF_TEMPLATE_REGEX, df.at[index, "subject_map_value"])
                df.at[index, "subject_references"] = references_list
                df.at[index, "subject_reference_count"] = len(references_list)
                df.at[index, "subject_references_template"] = re.sub(REF_TEMPLATE_REGEX, '([^\/]*)', df.at[index, "subject_map_value"]) + '$'
                
        match df.at[index, "predicate_map_type"]:
            case "http://w3id.org/rml/constant":
                df.at[index, "predicate_references"] = []
                df.at[index, "predicate_reference_count"] = 0

            case "http://w3id.org/rml/reference":
                df.at[index, "predicate_references"] = [df.at[index, "predicate_map_value"]]
                df.at[index, "predicate_reference_count"] = 1

            case "http://w3id.org/rml/template":
                references_list = re.findall(REF_TEMPLATE_REGEX, df.at[index, "predicate_map_value"])
                df.at[index, "predicate_references"] = references_list
                df.at[index, "predicate_reference_count"] = len(references_list)
                df.at[index, "predicate_references_template"] = re.sub(REF_TEMPLATE_REGEX, '([^\/]*)', df.at[index, "predicate_map_value"]) + '$'

        match df.at[index, "object_map_type"]:
            case "http://w3id.org/rml/constant":
                df.at[index, "object_references"] = []
                df.at[index, "object_reference_count"] = 0

            case "http://w3id.org/rml/reference":
                df.at[index, "object_references"] = [df.at[index, "object_map_value"]]
                df.at[index, "object_reference_count"] = 1

            case "http://w3id.org/rml/template":
                references_list = re.findall(REF_TEMPLATE_REGEX, df.at[index, "object_map_value"])
                df.at[index, "object_references"] = references_list
                df.at[index, "object_reference_count"] = len(references_list)
                df.at[index, "object_references_template"] = re.sub(REF_TEMPLATE_REGEX, '([^\/]*)', df.at[index, "object_map_value"]) + '$'

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

def generate_object_fragment(rule:pd.Series, subject_index:int) -> str|None:
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

def generate_subject_fragment(rule:pd.Series, subject_index:int) -> str|None:
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

def generate_query(mapping_rules, iterator_rules:pd.DataFrame) -> str|None:
    references = get_references(iterator_rules)
    if len(references) == 0:
        return None
    lines = []
    rule = iterator_rules.iloc[0]
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

def invert_source(mapping_rules:pd.DataFrame, source_rules:pd.DataFrame, endpoint:Endpoint):
    iterator_result = None
    for iterator, iterator_rules in source_rules.groupby('iterator', dropna=False):
        query = generate_query(mapping_rules, iterator_rules)
        inversion_logger.debug(query)
        if query is None:
            inversion_logger.warning("No query generated (no references found)")
            iterator_result = None
        else:
            iterator_result = endpoint.query(query)
    # generate template
    # fill template with values
    # we dont do any of this yet (PoC)
    if iterator_result is None:
        return None
    return iterator_result
    

def inversion(config_file: str|pathlib.Path):
    config = load_config_from_argument(config_file)
    mappings:pd.DataFrame
    mappings, _= retrieve_mappings(config)
    endpoint = EndpointFactory.create(config)
    insert_columns(mappings)
    for source, source_rules in mappings.groupby("logical_source_value"):
        generated_source = invert_source(mappings, source_rules, endpoint)

        # after this should really be in its own function
        if generated_source is None:
            generated_df = pd.DataFrame()
        else:
            generated_df = pd.read_csv(StringIO(generated_source))     # this is only for csv files
        with open(source, "r") as file:
            expected_source = pd.read_csv(file)

        inversion_logger.debug(generated_df)
        inversion_logger.debug(expected_source)
        if Validator.df_equals(generated_df, expected_source):
            inversion_logger.debug("Dataframes are equal")
            inversion_logger.info("Test passed")
        else:
            inversion_logger.debug("Dataframes are not equal")
            inversion_logger.info("Test failed")
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
        inversion_logger.info(f'Running test {row["RML id"]}, ({row["better RML id"]})')
        os.chdir(testcases_path / row["RML id"])
        try:
            inversion(MORPH_CONFIG)
        except Exception as e:
            inversion_logger.debug(e)
            inversion_logger.info("Test failed (exception: %s)", type(e).__name__)
        


def main():
    if os.path.exists("inversion.log"):
        os.remove("inversion.log")
    inversion_logger.setLevel(logging.DEBUG)
    inversion_logger.propagate = False
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    file_logger = logging.FileHandler("inversion.log")
    file_logger.setLevel(logging.DEBUG)
    file_logger.setFormatter(formatter)
    consolelogger = logging.StreamHandler()
    consolelogger.setLevel(logging.INFO)
    consolelogger.setFormatter(formatter)
    inversion_logger.addHandler(file_logger)
    inversion_logger.addHandler(consolelogger)
    # ignore morph_kgc logs
    warnings.simplefilter(action='ignore', category=FutureWarning)
    test_rml_test_cases()
    

if __name__ == '__main__':
    main()