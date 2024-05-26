import importlib.util
import sys
import pathlib
import pytest
from pytest_csv_params.decorator import csv_params
import pandas as pd
from urllib.parse import ParseResult, urlparse, unquote
import os
from io import StringIO
import json
import logging
import re
from rdflib.graph import Graph
import morph_kgc
import rdflib.compare

pytest_rml_test_cases_dir = pathlib.Path(__file__).parent
implementation_dir = pathlib.Path(__file__).parent.parent.parent
test_cases_path = implementation_dir / "rml-test-cases" / "test-cases"
inversion_path = implementation_dir / "poc_inversion.py"

spec = importlib.util.spec_from_file_location("inversion", inversion_path.absolute())
inversion = importlib.util.module_from_spec(spec)
sys.modules["inversion"] = inversion
spec.loader.exec_module(inversion)

MORPH_CONFIG_CONVERT = """
    [CONFIGURATION]
    # INPUT
    na_values=,#N/A,N/A,#N/A N/A,n/a,NA,<NA>,#NA,NULL,null,NaN,nan,None

    # OUTPUT
    output_file=output.nq
    output_dir=
    output_format=N-QUADS
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

MORPH_CONFIG_GENERATED = """
    [CONFIGURATION]
    # INPUT
    na_values=,#N/A,N/A,#N/A N/A,n/a,NA,<NA>,#NA,NULL,null,NaN,nan,None

    # OUTPUT
    output_file=output_generated.nq
    output_dir=
    output_format=N-QUADS
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
    mappings: mapping_test.ttl
"""

SOURCE_REGEX = 'rml:source "([0-z]*).([0-z]*)";'

def create_check_mapping():
    with open("mapping.ttl", "r") as file:
        mapping = file.read()
    # replace source match 1 with source_generated
    sources_list = []
    for match in re.finditer(SOURCE_REGEX, mapping):
        sources_list.append(match.group(1))
    sources_list = list(set(sources_list))
    for source in sources_list:
        mapping = mapping.replace(source, f"{source}_generated")
    with open("mapping_test.ttl", "w") as file:
        file.write(mapping)
    
class Validator:
    @staticmethod
    def url(x) -> bool:
        try:
            result: ParseResult = urlparse(x)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    @staticmethod
    def df_equals(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        # pure function
        df1 = df1.copy(deep=True)
        df2 = df2.copy(deep=True)
        # sort by columns and rows
        df1.sort_index(axis=1, inplace=True)
        df1.sort_values(by=list(df1.columns), inplace=True)
        df1.drop_duplicates(inplace=True)
        df1.index = range(df1.shape[0])
        df2.sort_index(axis=1, inplace=True)
        df2.sort_values(by=list(df2.columns), inplace=True)
        df2.drop_duplicates(inplace=True)
        df2.index = range(df2.shape[0])
        try:
            pd.testing.assert_frame_equal(df1, df2, check_like=False, check_dtype=False)
            return True
        except AssertionError:
            return False
    
    
    @staticmethod
    def _order_object(obj):
        """Order an object recursively
    
        Args:
            obj (object): any object

        Returns:
            obj: sorted object"""
        if isinstance(obj, dict):
            return sorted((k, Validator._order_object(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return sorted(Validator._order_object(x) for x in obj)
        else:
            return obj
    
    def _stringify_values(obj):
        """Stringify all values in an object recursively

        Args:
            obj (object): any object

        Returns:
            object: object with all values stringified
        """
        if isinstance(obj, dict):
            return {str(k): Validator._stringify_values(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Validator._stringify_values(x) for x in obj]
        else:
            return str(obj)
    
    @staticmethod
    def json_equals(json1: str, json2: str) -> bool:
        """Compare two jsons for equality, regardless of the order of the keys

        Args:
            json1 (str): first json as string
            json2 (str): second json as string
        
        Returns:
            bool: True if the jsons are equal, False otherwise
        """
        try:
            json_loaded1 = json.loads(json1)
            json_loaded2 = json.loads(json2)
        except json.JSONDecodeError:
            return False
        json_stringified1 = Validator._stringify_values(json_loaded1)
        json_stringified2 = Validator._stringify_values(json_loaded2)       
        json_ordered1 = Validator._order_object(json_stringified1)
        json_ordered2 = Validator._order_object(json_stringified2)
        return json_ordered1 == json_ordered2
    
    @staticmethod
    def graph_equals(graph1: Graph, graph2: Graph) -> bool:
        """Compare two graphs for equality

        Args:
            graph1_filename (str): first graph file
            graph2_filename (str): second graph file

        Returns:
            bool: True if the graphs are equal, False otherwise
        """
        return rdflib.compare.isomorphic(graph1, graph2)



csv_file = str(pytest_rml_test_cases_dir / "input_CSV.csv")
json_file = str(pytest_rml_test_cases_dir / "input_JSON.csv")

@csv_params(
    data_file= csv_file,
    header_renames={
        "RML id": "rml_id",
        "better RML id": "better_rml_id"
    },
    data_casts={
        "rml_id": str,
        "better_rml_id": str
    }
)
def test_csv_test_case(
    rml_id: str,
    better_rml_id: str
) -> None:
    logger:logging.Logger = inversion.get_logger()
    logger.debug(f"Running test case {rml_id} with better RML id {better_rml_id}")
    original_dir = os.getcwd()
    os.chdir(test_cases_path / rml_id)
    results = inversion.inversion(MORPH_CONFIG_CONVERT, rml_id)
    passed = 0
    for source, source_result in results.items():
        with open(source, "r") as file:
            expected_source = pd.read_csv(file)
        logger.debug("Generated: " + source_result)
        logger.debug("Original:" + expected_source.to_csv(index=False))
        try:
            source_result_df = pd.read_csv(StringIO(source_result))
        except Exception as e:
            source_result_df = pd.DataFrame()
        if Validator.df_equals(source_result_df, expected_source):
            logger.debug(f"Dataframes are equal for {source}")
            logger.debug("Test passed")
            passed += 1
        else:
            logger.debug(f"Dataframes are not equal for {source}")
            logger.debug("Test failed")
            file_name = source.split(".")[0]
            generated_source = f"{file_name}_generated.csv"
            source_result_df.to_csv(generated_source, index=False)         
    if passed == len(results):
        return
    
    create_check_mapping()
    morph_graph = morph_kgc.materialize(MORPH_CONFIG_GENERATED)
    with open("output_generated.nt", "w") as file:
        morph_triples = morph_graph.serialize(format="nt")
        file.write(morph_triples)
    original_graph = Graph()
    with open("output.nq", "r") as file:
        content = file.read()
        content = content.replace("E0", "")
        original_graph.parse(data=content, format="nquads")
    if morph_triples.strip().count("\n") != content.strip().count("\n"):
        logger.debug("Number of triples is different")
        logger.debug("Test failed")
        os.chdir(original_dir)
        pytest.fail(f"Test case {rml_id} failed")
    if Validator.graph_equals(morph_graph, original_graph):
        logger.debug("Graphs are equal")
        logger.debug("Test passed")
    else:
        logger.debug("Graphs are not equal")
        logger.debug("Test failed")
        os.chdir(original_dir)
        pytest.fail(f"Test case {rml_id} failed")
    os.chdir(original_dir)
    
@csv_params(
    data_file= json_file,
    header_renames={
        "RML id": "rml_id",
        "better RML id": "better_rml_id"
    },
    data_casts={
        "rml_id": str,
        "better_rml_id": str
    }
)
def test_json_test_case(
    rml_id: str,
    better_rml_id: str
) -> None:
    logger:logging.Logger = inversion.get_logger()
    logger.debug(f"Running test case {rml_id} with better RML id {better_rml_id}")
    original_dir = os.getcwd()
    os.chdir(test_cases_path / rml_id)
    try:
        results = inversion.inversion(MORPH_CONFIG_CONVERT, rml_id)
    except Exception as e:
        os.chdir(original_dir)
        pytest.fail(f"Test case {rml_id} failed with error: {e}")
    passed = 0
    for source, source_result in results.items():
        with open(source, "r") as file:
            expected_source = file.read()
        logger.debug("Generated: " + source_result)
        logger.debug("Original:" + expected_source)
        file_name = source.split(".")[0]
        generated_source = f"{file_name}_generated.json"
        with open(generated_source, "w") as file:
            file.write(source_result)
        if Validator.json_equals(source_result, expected_source):
            logger.debug(f"JSONs are equal for {source}")
            logger.debug("Test passed")
            passed += 1
        else:
            logger.debug(f"JSONs are not equal for {source}")
            logger.debug("Test failed, checking if the re-generated graph is equal to the original graph")
    if passed == len(results):
        return
            
    
    # If the JSONs are not equal, check if the re-generated graph is equal to the original graph
    create_check_mapping()
    morph_graph = morph_kgc.materialize(MORPH_CONFIG_GENERATED)
    with open("output_generated.nt", "w") as file:
        morph_triples = morph_graph.serialize(format="nt")
        file.write(morph_triples)
    original_graph = Graph()
    with open("output.nq", "r") as file:
        content = file.read()
        content = content.replace("E0", "")
        original_graph.parse(data=content, format="nquads")
    if morph_triples.strip().count("\n") != content.strip().count("\n"):
        logger.debug("Number of triples is different")
        logger.debug("Test failed")
        os.chdir(original_dir)
        pytest.fail(f"Test case {rml_id} failed")
    if Validator.graph_equals(morph_graph, original_graph):
        logger.debug("Graphs are equal")
        logger.debug("Test passed")
    else:
        logger.debug("Graphs are not equal")
        logger.debug("Test failed")
        os.chdir(original_dir)
        pytest.fail(f"Test case {rml_id} failed")
    os.chdir(original_dir)
            
    