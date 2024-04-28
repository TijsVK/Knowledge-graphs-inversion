import time

from morph_kgc.mapping.mapping_parser import retrieve_mappings
from morph_kgc.args_parser import load_config_from_argument
import pathlib
import json
import pandas as pd
import warnings
import os
import sys
pyrdf4j_path = pathlib.Path(__file__).parent / "pyrdf4j"
sys.path.append(str(pyrdf4j_path))
import pyrdf4j.rdf4j
import pyrdf4j.errors
from SPARQLWrapper import SPARQLWrapper, JSON
import re
import cProfile

MAPPINGFILENAME = "mapping.ttl"
OUTPUTFILENAME = "output.nq"

# debug flags
DEBUG = True  # if True, do not redirect stdout to file
PRINT_MAPPING_FILE = False
PRINT_TRIPLE_FILE = True
PRINT_OBJECT_IDS = False
PRINT_MAPPINGS = True
PRINT_FILTERING = True
PRINT_EXTRACTING = True
PRINT_SUBJECT_INFORMATION = True
PRINT_ROW_ADDING = True
PRINT_OUTPUT = True
PRINT_ID_QUERY_RESULTS = True

pyrdf4j.repo_types.REPO_TYPES = pyrdf4j.repo_types.REPO_TYPES + ["graphdb"]  # add graphdb to the list of repo types


def compare_dfs(df1: pd.DataFrame, df2: pd.DataFrame):
    # sort both dataframes by columns
    df1.sort_index(axis=1, inplace=True)
    df2.sort_index(axis=1, inplace=True)
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


get_all_subjects_query = """
SELECT DISTINCT ?s
WHERE {
    ?s ?p ?o
}
"""

MORPH_CONFIG = f"""
    [CONFIGURATION]
    # INPUT
    na_values=,#N/A,N/A,#N/A N/A,n/a,NA,<NA>,#NA,NULL,null,NaN,nan,None

    # OUTPUT
    output_file=output.nt
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

get_all_subjects_and_ids_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
select  distinct ?s ?id
where {
    ?s ?p ?o.
    optional{
        ?s <http://www.ontotext.com/owlim/entity#id> ?id.
    }
}
"""


def get_all_data_for_id_query(subject_id):
    return f"""
SELECT ?s ?p ?o
WHERE {{
    ?s ?p ?o.
    ?s <http://www.ontotext.com/owlim/entity#id> {subject_id}.
}}
"""


def insert_columns(df: pd.DataFrame, pure=False) -> pd.DataFrame:
    # add columns to end of dataframe, probably faster than inserting them in the middle but worse for overview
    # rules_df["subject_references"] = [[] for _ in range(rules_df.shape[0])]
    # rules_df["subject_reference_count"] = 0
    # rules_df["predicate_references"] = [[] for _ in range(rules_df.shape[0])]
    # rules_df["predicate_reference_count"] = 0
    # rules_df["object_references"] = [[] for _ in range(rules_df.shape[0])]
    # rules_df["object_reference_count"] = 0

    # add columns to dataframe at specific index
    # probably slower than adding them to the end but better for overview when printing rows
    if pure:
        df = df.copy(deep=True)  # do not modify original dataframe (pure function)
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


def add_data_to_repository(rdf4jconnector: pyrdf4j.rdf4j.RDF4J, sparql: SPARQLWrapper, repo_id: str) -> list:
    sparql.setQuery(get_all_subjects_and_ids_query)
    id_query_result = sparql.query().convert()["results"]["bindings"]
    orig_ids = [result["id"]["value"] for result in id_query_result]
    if PRINT_OBJECT_IDS:
        print(f"Original ids: {orig_ids}")
    with open(OUTPUTFILENAME, "r") as file:
        file_contents = file.read()
        if PRINT_TRIPLE_FILE:
            print(file_contents)
        rdf4jconnector.add_data_to_repo(repo_id, file_contents, "text/x-nquads")
        data_ids = [result["id"]["value"] for result in sparql.query().convert()["results"]["bindings"] if
                    result["id"]["value"] not in orig_ids]
        if PRINT_OBJECT_IDS:
            print(f"Data ids: {data_ids}")
    return data_ids


def setup_test(repo_id, rdf4jconnector: pyrdf4j.rdf4j.RDF4J):
    config = MORPH_CONFIG
    rdf4jconnector.drop_repository(repo_id, accept_not_exist=True)
    rdf4jconnector.create_repository(repo_id, accept_existing=True, repo_type='graphdb')
    sparql = SPARQLWrapper(f"http://localhost:7200/repositories/{repo_id}")
    sparql.setReturnFormat(JSON)
    data_ids = add_data_to_repository(rdf4jconnector, sparql, repo_id)
    loaded_config = load_config_from_argument(config)
    rules_df: pd.DataFrame
    rules_df, _ = retrieve_mappings(loaded_config)

    return rules_df, sparql, data_ids


def get_all_subjects_ids(sparql: SPARQLWrapper, data_ids: list):
    sparql.setQuery(get_all_subjects_and_ids_query)
    query_result = sparql.query().convert()
    subjects_ids_dict = query_result["results"]["bindings"]
    subjects_id_tuples = [(x["s"]["value"], x["id"]["value"]) for x in subjects_ids_dict if
                          x["id"]["value"] in data_ids]
    return subjects_id_tuples


def print_mappings_for_each_source(rules_df: pd.DataFrame):
    for source, rules in rules_df.groupby("logical_source_value"):
        print(f"{source}: {len(rules)} mapping(s)")
        for subjectTemplate, subjectRules in rules.groupby("subject_map_value"):
            print(f"{subjectTemplate}: {len(subjectRules)} mapping(s)")
            print(f"Rules:")
            for _, rule in subjectRules.iterrows():
                for key, value in rule.items():
                    print(f"\t{key}: {value}")
                print()


def construct_output_df(rules: pd.DataFrame):
    source = rules.at[rules.index[0], "logical_source_value"]
    columns = get_all_references_for_rules(rules)
    print(f"{source}: {len(columns)} reference(s): {columns}")

    output_source_df = pd.DataFrame(columns=list(columns))
    output_source_df.sort_index(axis=1, inplace=True)

    return output_source_df


def get_all_references_for_rules(rules: pd.DataFrame):
    references = set()
    for _, rule in rules.iterrows():
        for reference in rule["subject_references"]:
            references.add(reference)
        for reference in rule["predicate_references"]:
            references.add(reference)
        for reference in rule["object_references"]:
            references.add(reference)
    return references


def get_matching_templates(rules_df: pd.DataFrame, subject: str) -> set:
    matching_templates = set()
    for subjectTemplate, subjectRules in rules_df.groupby("subject_map_value"):
        if subjectRules.at[subjectRules.index[0], "subject_map_type"] == "http://w3id.org/rml/template":
            # match got swapped with search because of blank nodes
            # if this causes problems, swap them back and add a check for blank nodes
            if re.search(subjectRules.at[subjectRules.index[0], "subject_references_template"], subject):
                matching_templates.add(subjectTemplate)
    return matching_templates


def subject_template_matches_rule(rule: pd.Series, p_o_pairs: dict) -> bool:
    subject_template = rule["subject_map_value"]
    predicate_value = rule["predicate_map_value"]
    predicate_type = rule[
        "predicate_map_type"]
    object_value = rule["object_map_value"]
    object_type = rule["object_map_type"]
    object_template = rule["object_references_template"]

    if predicate_type == "http://w3id.org/rml/constant":
        if predicate_value in p_o_pairs.keys():
            if PRINT_FILTERING:
                print(f"Constant predicate {predicate_value} found in subject predicates. Checking object.")
            if object_type == "http://w3id.org/rml/constant":
                if object_value == p_o_pairs[predicate_value]:
                    if PRINT_FILTERING:
                        print(
                            f"Constant object {object_value} matches S-P-O ({subject_template} - {predicate_value} - {p_o_pairs[predicate_value]})")
                    return True
                else:
                    if PRINT_FILTERING:
                        print(
                            f"Object value {object_value} does not match S-P-O ({subject_template} - {predicate_value} - {p_o_pairs[predicate_value]})")
                    return False
            elif object_type == "http://w3id.org/rml/template":
                if re.match(object_template, p_o_pairs[predicate_value]):
                    if PRINT_FILTERING:
                        print(
                            f"Template object {object_value} matches subject {subject_template} - predicate {predicate_value} - object {p_o_pairs[predicate_value]}")
                    return True
                else:
                    if PRINT_FILTERING:
                        print(
                            f"Object value {object_value} does not match subject {subject_template} - predicate {predicate_value} - object {p_o_pairs[predicate_value]}")
                    return False
            elif object_type == "http://w3id.org/rml/reference":
                if PRINT_FILTERING:
                    print(
                        f"Reference object {object_value} found in subject {subject_template} - predicate {predicate_value} - object {p_o_pairs[predicate_value]} (nothing to check)")
                return True
        else:
            if PRINT_FILTERING:
                print(f"Predicate {rule['predicate_map_value']} not found in subject {subject_template}.")
            return False
    elif predicate_type == "http://w3id.org/rml/template":
        if PRINT_FILTERING:
            print(f"Template predicate {predicate_value} found in subject predicates. This is not supported yet.")
        return False
    else:
        raise ValueError(f"Predicate type {predicate_type} not supported.")
    return True


def filter_matching_templates(rules_df: pd.DataFrame, p_o_pairs: dict, matching_templates: set) -> set:
    new_matching_templates = set()
    for subject_template in matching_templates:
        subject_rules = rules_df[rules_df["subject_map_value"] == subject_template]
        for i, rule in subject_rules.iterrows():
            if subject_template_matches_rule(rule, p_o_pairs):
                continue
            else:
                break
        else:  # if all rules match
            new_matching_templates.add(subject_template)
    return new_matching_templates


# TODO: this could use the Strategy design pattern and be split into multiple functions
def get_and_apply_best_extraction_rule_for_reference(
        reference: str,
        rules: pd.DataFrame,
        subject: str,
        predicate_objects_dict: dict) -> str | None:
    # prefer reference rules
    for _, rule in rules.iterrows():
        if rule["subject_map_type"] == "http://w3id.org/rml/reference" and reference in rule["subject_references"]:
            if PRINT_EXTRACTING:
                print(f"Reference {reference} found in subject {rule['subject_map_value']} -> {subject}")
            return subject
        if rule["object_map_type"] == "http://w3id.org/rml/reference" and reference in rule["object_references"]:
            if PRINT_EXTRACTING:
                print(
                    f"Reference {reference} found in object {rule['object_map_value']} -> {predicate_objects_dict[rule['predicate_map_value']]}")
            return predicate_objects_dict[rule["predicate_map_value"]]

    # prefer single reference templates
    for _, rule in rules.iterrows():
        if (rule["subject_map_type"] == "http://w3id.org/rml/template"
                and rule["subject_reference_count"] == 1
                and reference in rule["subject_references"]):
            value = re.search(rule["subject_references_template"], subject).group(1)
            if PRINT_EXTRACTING:
                print(f"Reference {reference} found in subject {rule['subject_map_value']} -> {value}")
            return value
        if (rule["object_map_type"] == "http://w3id.org/rml/template"
                and rule["object_reference_count"] == 1
                and reference in rule["object_references"]):
            object_value = predicate_objects_dict[rule["predicate_map_value"]]
            value = re.search(rule["object_references_template"], object_value).group(1)
            if PRINT_EXTRACTING:
                print(f"Reference {reference} found in object {rule['object_map_value']} -> {value}")
            return value

    # if no single reference rule is found, try to find a template rule from which the reference can be extracted
    for _, rule in rules.iterrows():
        if rule["subject_map_type"] == "http://w3id.org/rml/template" and reference in rule["subject_references"]:
            regex_count = len(re.findall("\(\[\^\\\/\]\*\)", rule["subject_references_template"]))
            if regex_count == rule["subject_reference_count"]:
                match_index = rule["subject_references"].index(reference)
                value = re.search(rule["subject_references_template"], subject).group(match_index + 1)
                if PRINT_EXTRACTING:
                    print(f"Reference {reference} found in subject {rule['subject_map_value']} -> {value}")
                return value
        if rule["object_map_type"] == "http://w3id.org/rml/template" and reference in rule["object_references"]:
            regex_count = len(re.findall("\(\[\^\\\/\]\*\)", rule["object_references_template"]))
            if regex_count == rule["object_reference_count"]:
                match_index = rule["object_references"].index(reference)
                object_value = predicate_objects_dict[rule["predicate_map_value"]]
                value = re.search(rule["object_references_template"], object_value).group(match_index + 1)
                if PRINT_EXTRACTING:
                    print(f"Reference {reference} found in object {rule['object_map_value']} -> {value}")
                return value

    # unsplittable template rules are near impossible to use
    # any strategy using them would be quite complex

    return None


def run_test(row: pd.Series, rdf4j_connector: pyrdf4j.rdf4j.RDF4J):
    os.chdir(row["RML id"])
    print(row["RML id"], f"({row['better RML id']})", row["title"], row["purpose"], row["error expected?"])
    try:
        rules_df: pd.DataFrame
        sparql: SPARQLWrapper
        data_ids: list

        rules_df, sparql, data_ids = setup_test(row["better RML id"], rdf4j_connector)

        if PRINT_MAPPING_FILE:
            with open(MAPPINGFILENAME, "r") as file:
                print(file.read())

        insert_columns(rules_df)

        subjects_id_tuples = get_all_subjects_ids(sparql, data_ids)
        if PRINT_ID_QUERY_RESULTS:
            print(json.dumps(subjects_id_tuples, indent=4))

        if PRINT_MAPPINGS:
            print_mappings_for_each_source(rules_df)

        source_output_dfs_dict = {}

        for source, subject_rules in rules_df.groupby("logical_source_value"):
            output_source_df = construct_output_df(subject_rules)
            source_output_dfs_dict[source] = output_source_df

        for subject, subject_id in subjects_id_tuples:
            sparql.setQuery(get_all_data_for_id_query(subject_id))
            data_dict = sparql.query().convert()["results"]["bindings"]
            p_o_pairs = {x["p"]["value"]: x["o"]["value"] for x in data_dict}

            matching_templates = get_matching_templates(rules_df, subject)

            matching_templates = filter_matching_templates(rules_df, p_o_pairs, matching_templates)

            if PRINT_SUBJECT_INFORMATION:
                print()
                print(f"{subject}: {len(p_o_pairs)} P-O pair(s): {json.dumps(p_o_pairs, indent=4)}")
                print(
                    f"{subject}: {len(matching_templates)} matching unfiltered template(s): {list(matching_templates)}")
                print(f"Checking matches for {subject} with {len(matching_templates)} templates.")
                print(f"{subject}: {len(matching_templates)} matching filtered template(s): {list(matching_templates)}")

            if len(matching_templates) == 0:
                if PRINT_SUBJECT_INFORMATION:
                    print(f"No matching templates found for {subject}.")
                continue

            applicable_rules = rules_df[rules_df["subject_map_value"].isin(matching_templates)]
            if PRINT_SUBJECT_INFORMATION:
                print(f"{subject}: {len(applicable_rules)} applicable rule(s)")

            for source, rules in applicable_rules.groupby("logical_source_value"):
                output_source_df = source_output_dfs_dict[source]

                new_row = pd.Series(index=output_source_df.columns)

                for reference in output_source_df.columns:
                    new_row[reference] = get_and_apply_best_extraction_rule_for_reference(
                        reference, rules, subject, p_o_pairs)

                if PRINT_ROW_ADDING:
                    print(f"Adding row {new_row} to {source} output dataframe.")
                source_output_dfs_dict[source] = pd.concat([output_source_df, new_row.to_frame().transpose()], ignore_index=True)

        for source in source_output_dfs_dict.keys():
            source_output_dfs_dict[source].sort_index(axis=1, inplace=True)
            if PRINT_OUTPUT:
                print(f"Output for {source}:")
                print(source_output_dfs_dict[source])
                print()

            with open(source, "r") as file:
                expected_output_df = pd.read_csv(file)
                if PRINT_OUTPUT:
                    print(f"Expected output for {source}:")
                    print(expected_output_df)
                try:
                    compare_dfs(source_output_dfs_dict[source], expected_output_df)
                    if PRINT_OUTPUT:
                        print("Output matches expected output.")
                except (IndexError, ValueError) as e:
                    if PRINT_OUTPUT:
                        print("Output does not match expected output.")
                        print(type(e))
                        print(e)


    except pyrdf4j.errors.DataBaseNotReachable as e:
        print("You need to start a triplestore first.")
        quit()
    except ValueError as e:
        print(
            "Even though we filter the tests with error expected? == False, some tests still expectedly fail. Morhp-KGC isnt able to handle all the tests either, so those error too sometimes.")
        print(type(e))
        print(e)
        if str(e) != "Found an invalid graph termtype. Found values {'http://w3id.org/rml/BlankNode'}. Graph maps must be http://w3id.org/rml/IRI.":
            # morph-kgc uses build-in python errors, but catching all ValueErrors is not a good idea as it also catches errors that are not related to morph-kgc
            raise
    # rdf4j_connector.drop_repository(row["better RML id"], accept_not_exist=True)
    print('\n' * 5)
    os.chdir("..")


def main():
    rdf4jconnector = pyrdf4j.rdf4j.RDF4J(rdf4j_base="http://localhost:7200/")

    warnings.simplefilter(action='ignore', category=FutureWarning)

    current_folder = pathlib.Path(__file__).parent

    if not DEBUG:
        with open(current_folder / "basic-inversion-output.txt", "r") as file:
            with open(current_folder / "basic-inversion-old-output.txt", "w") as oldfile:
                oldfile.write(file.read())
        sys.stdout = open(current_folder / "basic-inversion-output.txt", "w")

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
        run_test(row, rdf4jconnector)


if __name__ == "__main__":
    folder_path = pathlib.Path(__file__).parent
    cProfile.run('main()', str(folder_path / "inversion.profile"))
