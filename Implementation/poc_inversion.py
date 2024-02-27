from abc import ABC, abstractmethod
import time
from typing import Any
from xml.dom.minidom import Document

import morph_kgc.config
from morph_kgc.mapping.mapping_parser import retrieve_mappings
from morph_kgc.args_parser import load_config_from_argument
from morph_kgc.constants import RML_IRI, RML_LITERAL, RML_BLANK_NODE, RML_TEMPLATE, RML_REFERENCE, RML_CONSTANT
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
from urllib.parse import ParseResult, urlparse, unquote
from io import StringIO
import hashlib
import logging
import jsonpath_ng

# region Constants

QUERY_MINIMAL = 0
QUERY_REDUCED = 1
QUERY_SIMPLE = 2
QUERY_FULL = 3

# endregion


# region Setup
pyrdf4j.repo_types.REPO_TYPES = pyrdf4j.repo_types.REPO_TYPES + [
    "graphdb"
]  # add graphdb to the list of repo types

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

REPO_ID = "inversion"
TRIPLESTORE_URL = f"http://localhost:7200/repositories/{REPO_ID}"

TEST_CASES_PATH = pathlib.Path(__file__).parent / "rml-test-cases" / "test-cases"
REF_TEMPLATE_REGEX = "{([^{}]*)}"


# endregion

# region classes

# region Abstract Base Classes

class Endpoint(ABC):
    @abstractmethod
    def query(self, query: str):
        raise NotImplementedError

class Selector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select(self):
        pass

# endregion

class IdGenerator:
    def __init__(self):
        self.counter = 0

    def get_id(self):
        self.counter += 1
        return self.counter

    def reset(self):
        self.counter = 0


class QueryTriple:
    def __init__(self, rule: pd.Series):
        self.rule = rule

    @property
    def references(self) -> set[str]:
        return set.union(
            self.subject_references,
            self.predicate_references,
            self.object_references
        )

    @property
    def uri_encoded_references(self) -> set[str]:
        object_type = self.rule["object_map_type"]
        if object_type == RML_TEMPLATE:
            return set.union(
                self.subject_references,
                self.predicate_references,
                self.object_references
            )
        return set.union(
            self.subject_references,
            self.predicate_references
        )

    @property
    def plain_references(self) -> set[str]:
        if self.rule["object_map_type"] == RML_REFERENCE:
            return set(
                self.object_references
            )
        return set()

    @property
    def subject_references(self) -> set[str]:
        return set(
            self.rule["subject_references"]
        )

    @property
    def predicate_references(self) -> set[str]:
        return set(
            self.rule["predicate_references"]
        )

    @property
    def object_references(self) -> set[str]:
        return set(
            self.rule["object_references"]
        )

    def generate(self, encoded_references:set[str], IdGenerator:IdGenerator) -> str|None:
        subject_reference_bytes = self.rule["subject_map_value"].encode("utf-8")
        subject_reference_hex = f"{subject_reference_bytes.hex()}"
        predicate = f'<{self.rule["predicate_map_value"]}>'
        object_map_value = self.rule["object_map_value"]
        object_map_type = self.rule["object_map_type"]
        if object_map_type == RML_CONSTANT:
            object_term_type = self.rule["object_termtype"]
            if object_term_type == RML_IRI:
                object_map_value = f'<{object_map_value}>'
            elif object_term_type == RML_BLANK_NODE:
                # raise NotImplementedError("Blank nodes are not implemented, and will not be implemented due to their nature.")
                return None
            return f"?{subject_reference_hex} {predicate} {object_map_value} ."

        object_reference_byte_string = object_map_value.encode("utf-8")
        object_reference_hex = object_reference_byte_string.hex()

        if object_map_type == RML_REFERENCE:    
            if object_map_value in encoded_references:
                lines = []
                plain_object_reference = f"{object_reference_hex}_plain_{IdGenerator.get_id()}"
                lines.append(f"OPTIONAL{{?{subject_reference_hex} {predicate} ?{plain_object_reference}}}")
                lines.append(f"OPTIONAL{{BIND(ENCODE_FOR_URI(?{plain_object_reference}) as ?{object_reference_hex}_encoded)}}")
                lines.append(f"FILTER(!BOUND(?{plain_object_reference}) || ENCODE_FOR_URI(?{plain_object_reference}) = ?{object_reference_hex}_encoded)")
                return "\n".join(lines)
            else:
                return f"OPTIONAL{{?{subject_reference_hex} {predicate} ?{object_reference_hex}}}"
            

        elif object_map_type == RML_TEMPLATE:
            lines = []
            full_template_reference = f"{object_reference_hex}_full_{IdGenerator.get_id()}"
            lines.append(f"OPTIONAL{{?{subject_reference_hex} {predicate} ?{full_template_reference}}}")
            lines.append(f"FILTER(!BOUND(?{full_template_reference}) || REGEX(STR(?{full_template_reference}), '{self.rule['object_references_template']}'))")
            evaluated_template = object_map_value
            current_reference = full_template_reference
            for reference in self.rule["object_references"]: # we cant use self.object_references here as the order is important (#TODO: refactor self.object_references)
                current_pre_string = evaluated_template.split("{", 1)[0]
                current_post_string = evaluated_template.split("}", 1)[1]
                next_pre_string = current_post_string.split("{", 1)[0]
                reference_byte_string = reference.encode("utf-8")
                reference_hex = reference_byte_string.hex()
                next_reference = f"{object_reference_hex}_slice_{IdGenerator.get_id()}"
                lines.append(f"OPTIONAL{{BIND(STRAFTER(STR(?{current_reference}), '{current_pre_string}') as ?{next_reference})}}")
                if current_post_string == "":
                    lines.append(f"OPTIONAL{{BIND(?{next_reference} as ?{reference_hex}_encoded)}}")
                else:
                    reference_placeholder = f"{reference_hex}_{IdGenerator.get_id()}"
                    lines.append(
                        f"BIND(STRBEFORE(STR(?{next_reference}), '{next_pre_string}') AS ?{reference_placeholder})"
                    )
                    lines.append(f"OPTIONAL{{BIND(?{reference_placeholder} as ?{reference_hex}_encoded)}}")
                    lines.append(f"FILTER(!BOUND(?{reference_hex}_encoded) || ?{reference_placeholder} = ?{reference_hex}_encoded)")

                evaluated_template = evaluated_template.split("}", 1)[1]
                current_reference = next_reference
            return "\n".join(lines)

class SubjectTriple(QueryTriple):
    def __init__(self, rule: pd.Series):
        super().__init__(rule)

    @property
    def uri_encoded_references(self) -> set[str]:
        return self.subject_references
    
    @property
    def plain_references(self) -> set[str]:
        return set()

    def generate(self, encoded_references: set[str], IdGenerator: IdGenerator) -> str | None:
        subject_map_value = self.rule["subject_map_value"]
        subject_map_type = self.rule["subject_map_type"]
        
        if subject_map_type != RML_TEMPLATE:
            return None
        
        subject_reference_byte_string = subject_map_value.encode("utf-8")
        subject_reference_hex = subject_reference_byte_string.hex()
        
        lines = []
        full_template_reference = f"?{subject_reference_hex}_full_{IdGenerator.get_id()}"
        lines.append(f"FILTER(REGEX(STR(?{full_template_reference}), '{self.rule['object_references_template']}'))")
        evaluated_template = subject_map_value
        current_reference = full_template_reference
        for reference in self.rule["subject_references"]: # we cant use self.object_references here as the order is important (#TODO: refactor self.object_references)
            current_pre_string = evaluated_template.split("{", 1)[0]
            current_post_string = evaluated_template.split("}", 1)[1]
            next_pre_string = current_post_string.split("{", 1)[0]
            reference_byte_string = reference.encode("utf-8")
            reference_hex = reference_byte_string.hex()
            next_reference = f"{subject_reference_hex}_slice_{IdGenerator.get_id()}"
            lines.append(f"OPTIONAL{{BIND(STRAFTER(STR(?{current_reference}), '{current_pre_string}') as ?{next_reference})}}")
            if current_post_string == "":
                lines.append(f"OPTIONAL{{BIND(?{next_reference} as ?{reference_hex})}}")
            else:
                reference_placeholder = f"{reference_hex}_{IdGenerator.get_id()}"
                lines.append(
                    f"BIND(STRBEFORE(STR(?{next_reference}), '{next_pre_string}') AS ?{reference_placeholder})"
                )
                lines.append(f"OPTIONAL{{BIND(?{reference_placeholder} as ?{reference_hex})}}")
                lines.append(f"FILTER(!BOUND(?{reference_hex}) || ?{reference_placeholder} = ?{reference_hex})")

            evaluated_template = evaluated_template.split("}", 1)[1]
            current_reference = next_reference

# region Selectors

class MinimalSelector(Selector):
    def select(self, triples: list[QueryTriple]):
        raise NotImplementedError

    def __str__(self):
        return "MinimalSelector"


class ReducedSelector(Selector):
    def select(self, triples: list[QueryTriple]):
        raise NotImplementedError

    def __str__(self):
        return "ReducedSelector"


class SimpleSelector(Selector):
    def select(self, triples: list[QueryTriple]):
        raise NotImplementedError

    def __str__(self):
        return "SimpleSelector"


class FullSelector(Selector):
    def select(self, triples: list[QueryTriple]):
        return triples

    def __str__(self):
        return "FullSelector"

class SelectorGenerator:
    @staticmethod
    def generate(selector: int) -> Selector:
        if selector == QUERY_MINIMAL:
            return MinimalSelector()
        elif selector == QUERY_REDUCED:
            return ReducedSelector()
        elif selector == QUERY_SIMPLE:
            return SimpleSelector()
        elif selector == QUERY_FULL:
            return FullSelector()
        else:
            raise ValueError(f"Unknown selector: {selector}")

# endregion

class Query:
    def __init__(self, triples: list[QueryTriple] = [], selector: Selector|int = QUERY_FULL):
        self.triples: list[QueryTriple] = triples
        if isinstance(selector, int):
            self.selector = SelectorGenerator.generate(selector)
        else:
            self.selector = selector
        self.idGenerator = IdGenerator()

    @property
    def references(self) -> list[str]:
        references = set()
        for triple in self.triples:
            references.update(triple.references)
        return list(references)

    @property
    def uri_encoded_references(self) -> list[str]:
        uri_encoded_references = set()
        for triple in self.triples:
            uri_encoded_references.update(triple.uri_encoded_references)
        return list(uri_encoded_references)
    
    @property
    def plain_references(self) -> list[str]:
        plain_references = set()
        for triple in self.triples:
            plain_references.update(triple.plain_references)
        return list(plain_references)

    @property
    def pure_references(self) -> list[str]:
        return [reference for reference in self.references if reference not in self.uri_encoded_references]

    def generate(self) -> str:
        # select triples using strategy
        inversion_logger.info(f"Selecting triples using {self.selector}")
        selected_triples = self.selector.select(self.triples)
        subject_count = len(set([triple.rule["subject_map_value"] for triple in selected_triples]))
        triple_count = len(selected_triples)

        all_references = self.references
        uri_encoded_references = self.uri_encoded_references
        plain_references = self.plain_references
        pure_references = self.pure_references
        
        if all_references == []:
            inversion_logger.warning("No references found, no query generated")
            return None

        inversion_logger.info(
            f"Selected {triple_count} triples with {subject_count} subjects having:\n\
                {len(uri_encoded_references)} URI encoded references: {uri_encoded_references}\n\
                {len(plain_references)} plain references: {plain_references}\n\
                {len(all_references)} all references: {all_references}")
        triple_strings = []
        for triple in selected_triples:
            triple_string = triple.generate(uri_encoded_references, self.idGenerator)
            if triple_string is not None:
                triple_strings.append(triple_string)
        
        select_part = "SELECT " + " ".join([f'?{reference.encode("utf-8").hex()}' for reference in pure_references] + [f'?{reference.encode("utf-8").hex()}_encoded' for reference in uri_encoded_references]) + " WHERE {"
        generated_query = select_part + "\n".join(triple_strings) + "}"
        return generated_query.replace("\\", "\\\\")

    def decode_dataframe(self, df: pd.DataFrame):
        df = df.copy(deep=True)
        for reference in self.uri_encoded_references:
            hex_reference = reference.encode("utf-8").hex()
            column = f"{hex_reference}_encoded"
            df[column] = df[column].apply(url_decode)
            df.rename(columns={column: reference}, inplace=True)
        for reference in self.pure_references:
            hex_reference = reference.encode("utf-8").hex()
            df.rename(columns={hex_reference: reference}, inplace=True)
        return df

    def execute_on_endpoint(self, endpoint: Endpoint) -> pd.DataFrame:
        generated_query = self.generate()
        csv_result = endpoint.query(generated_query)
        df = pd.read_csv(StringIO(csv_result))
        return self.decode_dataframe(df)


class RemoteEndpoint(Endpoint):
    def __init__(self, url: str):
        self._sparql = SPARQLWrapper(url)
        self._sparql.setReturnFormat(CSV)

    def query(self, query: str):
        self._sparql.setQuery(query)
        return self._sparql.query().convert().decode("utf-8")

    def __repr__(self):
        return f"RemoteSparqlEndpoint({self._sparql.endpoint})"


class LocalSparqlGraphStore(Endpoint):
    def __init__(self, url: str, delete_after_use: bool = False):
        data = open(url, "r", encoding="utf-8").read()
        self._repoid = hashlib.md5(data.encode("utf-8")).hexdigest()
        inversion_logger.debug(f"Creating repository: {self._repoid}")
        rdf4jconnector = pyrdf4j.rdf4j.RDF4J(rdf4j_base="http://localhost:7200/")
        rdf4jconnector.empty_repository(self._repoid)
        rdf4jconnector.create_repository(
            self._repoid, accept_existing=True, repo_type="graphdb"
        )
        rdf4jconnector.add_data_to_repo(self._repoid, data, "text/x-nquads")
        time.sleep(1)
        self._sparql = SPARQLWrapper(
            f"http://localhost:7200/repositories/{self._repoid}"
        )
        self._sparql.setReturnFormat(CSV)
        self.delete_after_use = delete_after_use

    def query(self, query: str) -> str:
        self._sparql.setQuery(query)
        query_result = self._sparql.query()
        converted: Any = query_result.convert()
        decoded = converted.decode("utf-8")
        return decoded

    def __del__(self):
        if self.delete_after_use:
            inversion_logger.debug(f"Dropping repository: {self._repoid}")
            rdf4jconnector = pyrdf4j.rdf4j.RDF4J(rdf4j_base="http://localhost:7200/")
            rdf4jconnector.drop_repository(self._repoid, accept_not_exist=True)

    def __repr__(self):
        return f"LocalSparqlGraphStore({self._repoid})"


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


class EndpointFactory:
    @classmethod
    def create(cls, config: morph_kgc.config.Config):
        url = config.get_output_file()
        return cls.create_from_url(url)

    @classmethod
    def create_from_url(cls, url: str):
        if Validator.url(url):
            return RemoteEndpoint(url)
        else:
            return LocalSparqlGraphStore(url)


class QueryExecutor:
    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint

    def execute(self, query: Query):
        pass


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
    df.insert(
        df.columns.get_loc("subject_map_value") + 1,
        "subject_references",
        [[] for _ in range(df.shape[0])],
    )
    df.insert(
        df.columns.get_loc("subject_map_value") + 1, "subject_references_template", None
    )
    df.insert(
        df.columns.get_loc("subject_references") + 1, "subject_reference_count", 0
    )
    df.insert(
        df.columns.get_loc("predicate_map_value") + 1,
        "predicate_references",
        [[] for _ in range(df.shape[0])],
    )
    df.insert(
        df.columns.get_loc("predicate_map_value") + 1,
        "predicate_references_template",
        None,
    )
    df.insert(
        df.columns.get_loc("predicate_references") + 1, "predicate_reference_count", 0
    )
    df.insert(
        df.columns.get_loc("object_map_value") + 1,
        "object_references",
        [[] for _ in range(df.shape[0])],
    )
    df.insert(
        df.columns.get_loc("object_map_value") + 1, "object_references_template", None
    )
    df.insert(df.columns.get_loc("object_references") + 1, "object_reference_count", 0)

    for index in df.index:
        match df.at[index, "subject_map_type"]:
            case "http://w3id.org/rml/constant":
                df.at[index, "subject_references"] = []
                df.at[index, "subject_reference_count"] = 0

            case "http://w3id.org/rml/reference":
                df.at[index, "subject_references"] = [df.at[index, "subject_map_value"]]
                df.at[index, "subject_reference_count"] = 1

            case "http://w3id.org/rml/template":
                references_list = re.findall(
                    REF_TEMPLATE_REGEX, df.at[index, "subject_map_value"]
                )
                df.at[index, "subject_references"] = references_list
                df.at[index, "subject_reference_count"] = len(references_list)
                df.at[index, "subject_references_template"] = (
                        re.sub(
                            REF_TEMPLATE_REGEX,
                            "([^\/]*)",
                            df.at[index, "subject_map_value"],
                        )
                        + "$"
                )

        match df.at[index, "predicate_map_type"]:
            case "http://w3id.org/rml/constant":
                df.at[index, "predicate_references"] = []
                df.at[index, "predicate_reference_count"] = 0

            case "http://w3id.org/rml/reference":
                df.at[index, "predicate_references"] = [
                    df.at[index, "predicate_map_value"]
                ]
                df.at[index, "predicate_reference_count"] = 1

            case "http://w3id.org/rml/template":
                references_list = re.findall(
                    REF_TEMPLATE_REGEX, df.at[index, "predicate_map_value"]
                )
                df.at[index, "predicate_references"] = references_list
                df.at[index, "predicate_reference_count"] = len(references_list)
                df.at[index, "predicate_references_template"] = (
                        re.sub(
                            REF_TEMPLATE_REGEX,
                            "([^\/]*)",
                            df.at[index, "predicate_map_value"],
                        )
                        + "$"
                )

        match df.at[index, "object_map_type"]:
            case "http://w3id.org/rml/constant":
                df.at[index, "object_references"] = []
                df.at[index, "object_reference_count"] = 0

            case "http://w3id.org/rml/reference":
                df.at[index, "object_references"] = [df.at[index, "object_map_value"]]
                df.at[index, "object_reference_count"] = 1

            case "http://w3id.org/rml/template":
                references_list = re.findall(
                    REF_TEMPLATE_REGEX, df.at[index, "object_map_value"]
                )
                df.at[index, "object_references"] = references_list
                df.at[index, "object_reference_count"] = len(references_list)
                df.at[index, "object_references_template"] = (
                        re.sub(
                            REF_TEMPLATE_REGEX, "([^\/]*)", df.at[index, "object_map_value"]
                        )
                        + "$"
                )

            case "http://w3id.org/rml/parentTriplesMap":
                df.at[index, "object_references"] = [
                    list(
                        json.loads(
                            df.at[index, "object_join_conditions"].replace("'", '"')
                        ).values()
                    )[0]["child_value"]
                ]
                df.at[index, "object_reference_count"] = 1

    return df


def get_references(rules: pd.DataFrame) -> set:
    references = set()
    for _, rule in rules.iterrows():
        for reference in rule["subject_references"]:
            references.add(reference)
        for reference in rule["predicate_references"]:
            references.add(reference)
        for reference in rule["object_references"]:
            references.add(reference)
    return references

def retrieve_data(
        mapping_rules: pd.DataFrame, source_rules: pd.DataFrame, endpoint: Endpoint
) -> pd.DataFrame | None:
    inversion_logger.debug(f"Processing source {source_rules.iloc[0]['logical_source_value']}")
    for _, rule in source_rules.iterrows():
        for key, value in rule.items():
            inversion_logger.debug(f"{key}: {value}")
    iterator_result:dict = {}
    for iterator, iterator_rules in source_rules.groupby("iterator", dropna=False):
        inversion_logger.debug(f"Processing iterator {iterator}")
        triples:list[QueryTriple] = []
        for _, rule in iterator_rules.iterrows():
            triples.append(QueryTriple(rule))
        for subject, subject_rules in iterator_rules.groupby("subject_map_value", dropna=False):
            triples.append(SubjectTriple(subject_rules.iloc[0]))
        query = Query(triples)
        generated_query = query.generate()
        # query = generate_query(mapping_rules, iterator_rules)
        inversion_logger.debug(query)
        if generated_query is None:
            inversion_logger.warning("No query generated (no references found)")
        else:
            inversion_logger.debug(generated_query)
            try:
                result = endpoint.query(generated_query)
                df = pd.read_csv(StringIO(result))
                for _, row in df.iterrows():
                    inversion_logger.debug(row)
                decoded_df = query.decode_dataframe(df)
                for _, row in decoded_df.iterrows():
                    inversion_logger.debug(row)
                iterator_result[iterator] = decoded_df
            except Exception as e:
                inversion_logger.warning(f"Error while querying endpoint: {e}")
                iterator_result[iterator] = None
    # join iterator results to get a source result
    # this should be done with some kind of join (inner/outer/left/right)
    # alternatively, we could return all the iterator results and let the template engine handle it
    # this could be more efficient too as a join would effectively duplicate the data for a shared parent
    # maybe even better would be to do this join server side with the query

    if len(iterator_result) == 0:
        return None
    else:
        return list(iterator_result.values())[0] # for now, just return the first iterators result


def generate_template(source_rules: pd.DataFrame) -> str:
    source_type = source_rules.iloc[0]["source_type"]
    inversion_logger.info(f"Generating template for source type {source_type}")

    if source_type == "JSON":
        for iterator, iterator_rules in source_rules.groupby("iterator", dropna=False):
            iterator_root = jsonpath_ng.parse(iterator)
            # accumulate all the references
            references = set()
            for _, rule in iterator_rules.iterrows():
                references.update(rule["subject_references"])
                references.update(rule["predicate_references"])
                references.update(rule["object_references"])
            inversion_logger.debug(f"References: {references}")


def inversion(config_file: str | pathlib.Path):
    config = load_config_from_argument(config_file)
    mappings: pd.DataFrame
    mappings, _ = retrieve_mappings(config)
    endpoint = EndpointFactory.create(config)
    insert_columns(mappings)
    results = {}
    for source, source_rules in mappings.groupby("logical_source_value"):
        source_data = retrieve_data(mappings, source_rules, endpoint)
        if source_data is None:
            inversion_logger.warning(f"No data generated for {source}")
            continue
        if source_rules.iloc[0]["source_type"] == "CSV":
            results[source] = source_data.to_csv(index=False)
        else:
            inversion_logger.warning(f"Source type {source_rules.iloc[0]['source_type']} not supported yet")
        template = generate_template(source_rules)
    return results
    


def url_decode(url):
    try:
        # check if url is a string
        if not isinstance(url, str):
            return url
        decoded_url = unquote(url)
        return decoded_url
    except Exception as e:
        # Handle invalid URLs or other decoding errors
        return url

def rml_test_cases():
    bad_tests = ["4a", "16a", "18a", "20a", "21a", "22a", "23a", "24a", "26a", "27a", "28a", "36a", "37a", "40a", "41a", "42a", "56a", "57a", "58a", "59a"]
    original_path = os.getcwd()
    os.chdir(TEST_CASES_PATH)
    this_file_path = pathlib.Path(__file__).resolve()
    implementation_dir = this_file_path.parent
    metadata_path = implementation_dir / "rml-test-cases" / "metadata.csv"
    testcases_path = implementation_dir / "rml-test-cases" / "test-cases"

    with open(metadata_path, "r") as file:
        tests_df: pd.DataFrame = pd.read_csv(file)

    tests_with_output = tests_df[tests_df["error expected?"] == False]
    # only CSV tests for now
    tests_with_output = tests_with_output[tests_with_output["data format"] == "CSV"]

    os.chdir(testcases_path)
    for _, row in tests_with_output.iterrows():
        if row["better RML id"] in bad_tests:
            inversion_logger.info(f'Skipping test {row["RML id"]}, ({row["better RML id"]})')
            continue
        inversion_logger.info(f'Running test {row["RML id"]}, ({row["better RML id"]})')
        os.chdir(testcases_path / row["RML id"])
        try:
            results = inversion(MORPH_CONFIG)
            for source, source_result in results.items():
                with open(source, "r") as file:
                    expected_source = pd.read_csv(file)
                inversion_logger.debug("Generated: " + source_result)
                inversion_logger.debug("Original:" + expected_source.to_csv(index=False))
                source_result_df = pd.read_csv(StringIO(source_result))
                if Validator.df_equals(source_result_df, expected_source):
                    inversion_logger.info(f"Dataframes are equal for {source}")
                    inversion_logger.info("Test passed")
                else:
                    inversion_logger.info(f"Dataframes are not equal for {source}")
                    inversion_logger.info("Test failed")
        except ValueError as e:
            inversion_logger.debug(e)
            inversion_logger.info("Test failed (exception: %s - %s)", type(e).__name__, e)
    os.chdir(original_path)
        

def run_tests():
    rml_test_cases()


def main():
    if os.path.exists("inversion.log"):
        # copy to inversion.log.old
        try:
            os.remove("inversion.log.old")
        except FileNotFoundError:
            pass
        os.rename("inversion.log", "inversion.log.old")
    inversion_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    file_logger = logging.FileHandler("inversion.log")
    file_logger.setLevel(logging.DEBUG)
    file_logger.setFormatter(formatter)
    inversion_logger.addHandler(file_logger)
    inversion_logger.setLevel(logging.DEBUG)
    inversion_logger.propagate = False
    consolelogger = logging.StreamHandler()
    consolelogger.setLevel(logging.INFO)
    consolelogger.setFormatter(formatter)
    inversion_logger.addHandler(consolelogger)
    # ignore morph_kgc FutureWarning logs
    warnings.simplefilter(action="ignore", category=FutureWarning)

    run_tests()

def test():
    this_file_path = pathlib.Path(__file__).resolve()
    implementation_dir = this_file_path.parent
    metadata_path = implementation_dir / "rml-test-cases" / "metadata.csv"
    testcases_path = implementation_dir / "rml-test-cases" / "test-cases"


    expr:jsonpath_ng.JSONPath = jsonpath_ng.parse("$.['students', 'teachers'][*]['Name', 'ID']")
    print(expr.__str__())
    print(expr.__repr__())
    students_json_string = """{
        "students": [{
            "ID": 10,
            "Name":"Venus"
        },
        {
            "ID": 11,
            "Name":"Luna"
        },
        {
            "ID": 12,
            "Name":"Mars"
        }]
    }"""
    students_json = json.loads(students_json_string)
    print([match.value for match in expr.find(students_json)])

if __name__ == "__main__":
    main()
