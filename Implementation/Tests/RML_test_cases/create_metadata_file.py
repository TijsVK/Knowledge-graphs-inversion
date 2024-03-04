import pathlib
import pandas as pd

pytest_rml_test_cases_dir = pathlib.Path(__file__).parent
implementation_dir = pathlib.Path(__file__).parent.parent.parent
test_cases_path = implementation_dir / "rml-test-cases" / "test-cases"
metadata_file = implementation_dir / "rml-test-cases" / "metadata.csv"
inversion_path = implementation_dir / "poc_inversion.py"

with open(metadata_file, "r") as file:
    tests_df: pd.DataFrame = pd.read_csv(file)

tests_with_output = tests_df[tests_df["error expected?"] == False]

additional_failed_tests = ["28a"]

tests_with_output = tests_with_output[~tests_with_output["better RML id"].isin(additional_failed_tests)]

bad_tests = {
    "CSV": ["4a", "16a", "18a", "20a", "21a", "22a", "23a", "24a", "26a", "27a", "28a", "31a", "36a", "37a", "40a", "41a", "42a", "56a", "57a", "58a", "59a"],
    "JSON": [],
    "XML": [],
    "RDF": [],
    "table": []
}

for data_format, tests in tests_with_output.groupby("data format"):
    tests = tests[~tests["better RML id"].isin(bad_tests[data_format])]
    # only keep "better RML id" and "RML id" columns
    tests = tests[["better RML id", "RML id"]]
    with open(pytest_rml_test_cases_dir / f"input_{data_format}.csv", "w") as file:
        tests.to_csv(file, index=False, lineterminator="\n")