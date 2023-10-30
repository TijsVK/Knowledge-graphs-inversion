import pathlib
import json
import pandas as pd
import time

metadatapath = pathlib.Path("Implementation/rml-test-cases/metadata.csv")
assert metadatapath.exists(), "File \"Implementation/rml-test-cases/metadata.csv\" does not exist"
testcasesbasepath = pathlib.Path("Implementation/rml-test-cases/test-cases")
assert testcasesbasepath.exists(), "Directory \"Implementation/rml-test-cases/test-cases\" does not exist"
mappingfilename = "mapping.ttl"
outputfilename = "output.nq"

with open(metadatapath, "r") as file:
    df:pd.DataFrame = pd.read_csv(file)

print(f"Total tests: {len(df.index)}")

table_tests = df[df["data format"] == "table"]
csv_style_tests = df[df["reference formulation"] == "CSV"]
csv_tests = df[df["data format"] == "CSV"]
print(f"Table tests: {len(table_tests.index)}")
print(f"CSV-style tests: {len(csv_style_tests.index)}")
print(f"CSV tests: {len(csv_tests.index)}")

table_tests_with_output = table_tests[table_tests["error expected?"] == False]
csv_style_tests_with_output = csv_style_tests[csv_style_tests["error expected?"] == False]
print(f"Table tests with output: {len(table_tests_with_output.index)}")
print(f"CSV-style tests with output: {len(csv_style_tests_with_output.index)}")


counts = {0: 0,
          1: 0,
          2: 0,
          3: 0}
for _, row in df.iterrows():
    directory = row["RML id"]
    print(row["RML id"], row["better RML id"], row["title"], row["purpose"])
    with open(testcasesbasepath / directory / mappingfilename, "r") as file:
        mapping = file.read()
        TriplesMapCount = mapping.count("a rr:TriplesMap;")
        if TriplesMapCount not in counts:
            counts[TriplesMapCount] = 1
        else:
            counts[TriplesMapCount] += 1
        # if TriplesMapCount == 0:
        #     print(row["RML id"], row["title"], row["purpose"])
for key, value in counts.items():
    print(f"{value} tests with {key} TriplesMaps")
    
print(df["reference formulation"].value_counts())
print(df["data format"].value_counts())