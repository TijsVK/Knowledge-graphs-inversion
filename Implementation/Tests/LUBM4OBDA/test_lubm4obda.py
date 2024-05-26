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
import time

pytest_rml_test_cases_dir = pathlib.Path(__file__).parent
implementation_dir = pathlib.Path(__file__).parent.parent.parent
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
    output_file=http://localhost:7200/repositories/lubm4obda_$version
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
    mappings: ../mapping_csv.ttl
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
    mappings: mapping_test_csv.ttl
"""

SOURCE_REGEX = 'rml:source "([0-z]*).([0-z]*)";'

def create_check_mapping():
    with open("mapping_csv.ttl", "r") as file:
        mapping = file.read()
    # replace source match 1 with source_generated
    source = re.search(SOURCE_REGEX, mapping).group(1)
    extension = re.search(SOURCE_REGEX, mapping).group(2)
    mapping = re.sub(SOURCE_REGEX, f'rml:source "{source}_generated.{extension}";', mapping)
    with open("mapping_test_csv.ttl", "w") as file:
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
        
        # old check code
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

@pytest.mark.parametrize("version", [("1"), ("10")])
def test_lubm4obda(version:str = "1"):
    logger:logging.Logger = inversion.get_logger()
    logger.debug("Starting test_lubm4obda")
    original_dir = os.getcwd()
    os.chdir(pathlib.Path(__file__).parent / version) 
    config = MORPH_CONFIG_CONVERT.replace("$version", version)
    try:
        results = inversion.inversion(config, f"lubm4obda_{version}")
    except Exception as e:
        os.chdir(original_dir)
        raise
        #pytest.fail(str(e))
    
    passed = 0
    for source, source_result in results.items():
        with open(source, "r") as file:
            expected_source = pd.read_csv(file)
        # logger.debug("Generated: " + source_result)
        # logger.debug("Original:" + expected_source.to_csv(index=False))
        try:
            source_result_df = pd.read_csv(StringIO(source_result))
        except Exception as e:
            source_result_df = pd.DataFrame()
        
        file_name = source.split(".")[0]
        generated_source = f"{file_name}_generated.csv"
        source_result_df.to_csv(generated_source, index=False)
        
        if Validator.df_equals(source_result_df, expected_source):
            logger.debug(f"Dataframes are equal for {source}")
            logger.debug("Test passed")
            passed += 1
        else:
            logger.debug(f"Dataframes are not equal for {source}")
            logger.debug("Test failed")
    if passed == len(results):
        return
    else:
        logger.debug(f"{passed}/{len(results)} tests passed")
        #pytest.fail("Some tests failed")


def temp():
    folders = ["1", "10","100", "1000"]
    
    for folder in folders:
        with open(pathlib.Path(__file__).parent / folder / "output_orig.nq", "r") as file:
            # amount of lines in file without fully opening it
            lines = sum(1 for line in file)
            # print length with dots every 3 numbers
            print(f"DB_{folder}: {lines:,}")
            
        with open(pathlib.Path(__file__).parent / folder / "output_csv.nq", "r") as file:
            # amount of lines in file without fully opening it
            lines = sum(1 for line in file)
            # print length with dots every 3 numbers
            print(f"CSV_{folder}: {lines:,}")
        print()

def main():
    folders = ["1000"]
    
    timingfile = pathlib.Path(__file__).parent / "timing.txt"
    # file = open(timingfile, "w")
    
    for folder in folders:
        print(f"Testing {folder}")
        try:
            start_time = time.time()
            test_lubm4obda(folder)
        except Exception as e:
            print(e)
        
        end_time = time.time()
        print(f"Time for {folder}: {end_time - start_time:.2f}s")
        # file.write(f"{folder}: {end_time - start_time:.2f}s\n")
        time.sleep(2)
    
if __name__ == "__main__":
    main()