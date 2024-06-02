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

# bad_tests = []

# # 4 series tests use the blank node ID to store values and are thus excluded
# bad_tests += ["4a", "4b", "4c", "4d"]
# # 28a can not be loaded by morph-kgc
# bad_tests += ["28a"]
# # 42a uses a blank node to store values... 
# # This test case specifically could be fixed (the subject is shared between two sources, and the data is stored in one of the triple maps)
# # Very low priority though
# bad_tests += ["42a"]

# non_bad_tests = tests_with_output[~tests_with_output["better RML id"].isin(bad_tests)]

for data_format, tests in tests_with_output.groupby("data format"):
    # only keep "better RML id" and "RML id" columns
    tests = tests[["better RML id", "RML id"]]
    with open(pytest_rml_test_cases_dir / f"input_{data_format}.csv", "w") as file:
        tests.to_csv(file, index=False, lineterminator="\n")