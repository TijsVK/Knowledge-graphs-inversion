import importlib.util
import sys
import pathlib
import pytest
from pytest_csv_params.decorator import csv_params
import pandas as pd
from urllib.parse import ParseResult, urlparse, unquote
import os
from io import StringIO

pytest_rml_test_cases_dir = pathlib.Path(__file__).parent
implementation_dir = pathlib.Path(__file__).parent.parent.parent
test_cases_path = implementation_dir / "rml-test-cases" / "test-cases"
inversion_path = implementation_dir / "poc_inversion.py"

spec = importlib.util.spec_from_file_location("inversion", inversion_path.absolute())
inversion = importlib.util.module_from_spec(spec)
sys.modules["inversion"] = inversion
spec.loader.exec_module(inversion)

MORPH_CONFIG = """
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
        df2.sort_index(axis=1, inplace=True)
        df2.sort_values(by=list(df2.columns), inplace=True)
        df2.drop_duplicates(inplace=True)
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



csv_file = str(pytest_rml_test_cases_dir / "input_CSV.csv")
print(csv_file)

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
    print(f"Running test case {rml_id} with better RML id {better_rml_id}")
    original_dir = os.getcwd()
    os.chdir(test_cases_path / rml_id)
    results = inversion.inversion(MORPH_CONFIG)
    for source, source_result in results.items():
        with open(source, "r") as file:
            expected_source = pd.read_csv(file)
        print("Generated: " + source_result)
        print("Original:" + expected_source.to_csv(index=False))
        source_result_df = pd.read_csv(StringIO(source_result))
        if Validator.df_equals(source_result_df, expected_source):
            print(f"Dataframes are equal for {source}")
            print("Test passed")
        else:
            print(f"Dataframes are not equal for {source}")
            print("Test failed")
            os.chdir(original_dir)
            pytest.fail(f"Test case {rml_id} failed")
    os.chdir(original_dir)