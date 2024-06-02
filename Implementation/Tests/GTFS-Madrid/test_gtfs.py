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
from deepdiff import DeepDiff
from pprint import pprint

pytest_rml_test_cases_dir = pathlib.Path(__file__).parent
# implementation_dir = pathlib.Path(__file__).parent.parent
implementation_dir = pathlib.Path("C:\Github\Knowledge-graphs-inversion\Implementation")
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
    output_file=http://localhost:7200/repositories/gtfs_$version
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
    mappings: ../mapping.ttl
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
        diff = DeepDiff(json_loaded1, json_loaded2, ignore_order=True)
        # pprint(diff.get_stats())
        if not diff:
            return True
        else:
            return False
        
    
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
def test_gtfs(version:str = "1", type:str = "csv"):
    logger:logging.Logger = inversion.get_logger()
    timingfile = pathlib.Path(__file__).parent / "timing.txt"
    logger.debug(f"Testing {type}-{version}")
    start_time = time.time()
    logger.debug("Starting test_gtfs")
    original_dir = os.getcwd()
    os.chdir(pathlib.Path(__file__).parent / type / version) 
    config = MORPH_CONFIG_CONVERT.replace("$version", version)
    try:
        results = inversion.inversion(config, f"gtfs_{type}_{version}")
    except Exception as e:
        os.chdir(original_dir)
        raise
        #pytest.fail(str(e))
    
    end_time = time.time()
    logger.debug(f"Time for {type}-{version}: {end_time - start_time:.2f}s")
    with open(timingfile, "a") as file:
        file.write(f"{type}-{version}: {end_time - start_time:.2f}s\n")

    passed = 0
    for source, source_result in results.items():
        file_name = source.split(".")[0]
        generated_source = f"{file_name}_generated.{type}"
        with open(generated_source, "w") as file:
            file.write(source_result)
        
        continue
        
        with open(source, "r") as file:
            if type == "csv":
                expected_source = pd.read_csv(file)
            else:
                expected_source = file.read()
        # logger.debug("Generated: " + source_result)
        # logger.debug("Original:" + expected_source.to_csv(index=False))
        if type == "csv":
            try:
                source_result_df = pd.read_csv(StringIO(source_result))
            except Exception as e:
                source_result_df = pd.DataFrame()
        
        if type == "csv":
            source_result_df.to_csv(generated_source, index=False)
            
            if Validator.df_equals(source_result_df, expected_source):
                logger.debug(f"Dataframes are equal for {source}")
                logger.debug("Test passed")
                passed += 1
            else:
                logger.debug(f"Dataframes are not equal for {source}")
                logger.debug("Test failed")
        else:
            with open(generated_source, "w") as file:
                file.write(source_result)
            if Validator.json_equals(source_result, expected_source):
                logger.debug(f"Jsons are equal for {source}")
                logger.debug("Test passed")
                passed += 1
            else:
                logger.debug(f"Jsons are not equal for {source}")
                logger.debug("Test failed")
                
    if passed == len(results):
        return
    else:
        logger.debug(f"{passed}/{len(results)} tests passed")
        #pytest.fail("Some tests failed")

def main():
    types = ["json"]
    folders = ["100"]
    
    for type in types:
        for folder in folders:
            test_gtfs(folder, type)
            


def test_generated_json():
    for version in ["1", "10", "100"]:
        os.chdir(pathlib.Path(__file__).parent / "json" / version)
        
        for source in ["AGENCY.json", "CALENDAR.json", "CALENDAR_DATES.json", "FEED_INFO.json", "FREQUENCIES.json", "ROUTES.json", "SHAPES.json", "STOP_TIMES.json", "STOPS.json", "TRIPS.json"]:
            with open(source, "r") as file:
                expected_source = file.read()
            with open(f"{source.split('.')[0]}_generated.json", "r") as file:
                source_result = file.read()
            assert Validator.json_equals(source_result, expected_source)

if __name__ == "__main__":
    main()