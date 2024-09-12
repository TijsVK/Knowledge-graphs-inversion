from abc import ABC, abstractmethod
import functools
import time
from typing import Any, Dict

import morph_kgc.config
from morph_kgc.mapping.mapping_parser import retrieve_mappings
from morph_kgc.args_parser import load_config_from_argument
from morph_kgc.constants import RML_IRI, RML_LITERAL, RML_BLANK_NODE, RML_TEMPLATE, RML_REFERENCE, RML_CONSTANT, RML_PARENT_TRIPLES_MAP
import pathlib
import json
import pandas as pd
import warnings
import os
import pyrdf4j.rdf4j
import pyrdf4j.errors
import pyrdf4j.repo_types
from SPARQLWrapper import SPARQLWrapper, CSV
import re
from urllib.parse import ParseResult, urlparse, unquote
from io import StringIO
import hashlib
import logging
import jsonpath_ng
from typing import Self  

# region Constants

QUERY_MINIMAL = 0
QUERY_REDUCED = 1
QUERY_FULL = 2

TEST_LOG_FOLDER = pathlib.Path(__file__).parent / "individual-logs"

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

TEST_CASES_PATH = pathlib.Path(__file__).parent / "rml-test-cases" / "test-cases"
REF_TEMPLATE_REGEX = "{([^{}]*)}"


# endregion

# region Abstract Base Classes

class Endpoint(ABC):
    @abstractmethod
    def query(self, query: str):
        raise NotImplementedError

class Triple(ABC):
    @abstractmethod
    def generate(self) -> str:
        raise NotImplementedError

class Selector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select(self) -> list[Triple]:
        pass
    
class Node(ABC):  
    def find(self, key: str) -> Self|None:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def path(self) -> str:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def parent_path(self) -> str:
        raise NotImplementedError
        
    @abstractmethod
    def to_template(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def fill(self, data: pd.DataFrame) -> str:
        raise NotImplementedError

class Template(ABC):
    @abstractmethod
    def create_template(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def fill_data(self, data: pd.DataFrame, source_name: str) -> str:
        raise NotImplementedError

# endregion

# region Utilities
class IdGenerator:
    def __init__(self):
        self.counter = 0

    def get_id(self):
        self.counter += 1
        return self.counter

    def reset(self):
        self.counter = 0

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

class Identifier:
    @staticmethod
    def generate_plain_identifier(rule: pd.Series, value: str) -> str | None:
        source_type:str = rule["source_type"]
        object_identifier:str
        if source_type == "CSV":
            object_identifier = value
        elif source_type == "JSON":
            try:
                object_identifier = JSONPathFunctions.extend_string_path(rule["iterator"], value)
            except Exception as e:
                return value
        elif source_type == "RDB":
            table_name = rule.get("logical_source_value", "")
            object_identifier = f"{table_name}.{value}"                
        else:
            inversion_logger.error(f"Unsupported source type: {source_type}")
            return None
        return object_identifier
    
class Codex:
    def __init__(self):
        self.codex: dict[str, str] = {}
        self.subjects: set[str] = set()
        self.idGenerator = IdGenerator()
    
    def get_id(self, key: str) -> str:
        if key in self.codex.keys():
            return self.codex[key]
        else:
            self.codex[key] = str(self.idGenerator.get_id())
            return self.codex[key]
        
    def get_id_and_is_bound(self, key: str) -> tuple[str, bool]:
        is_bound = key in self.codex.keys()
        return self.get_id(key), is_bound
    
# endregion

# region Triples

class QueryTriple(Triple):
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
            [Identifier.generate_plain_identifier(self.rule, value) for value in self.rule["subject_references"]]
        )

    @property
    def predicate_references(self) -> set[str]:
        return set(
            [Identifier.generate_plain_identifier(self.rule, value) for value in self.rule["predicate_references"]]
        )

    @property
    def object_references(self) -> set[str]:
        return set(
            [Identifier.generate_plain_identifier(self.rule, value) for value in self.rule["object_references"]]
        )

    def generate(self, encoded_references:set[str], IdGenerator:IdGenerator, codex: Codex, all_mapping_rules: pd.DataFrame) -> str|None:
        subject_reference = codex.get_id(self.rule["subject_map_value"])
        predicate = f'<{self.rule["predicate_map_value"]}>'
        object_map_value = self.rule["object_map_value"]
        object_map_type = self.rule["object_map_type"]
        object_references_template = self.rule["object_references_template"]
        
        
        
        if object_map_type == RML_CONSTANT:
            object_term_type = self.rule["object_termtype"]
            if object_term_type == RML_IRI:
                object_map_value = f'<{object_map_value}>'
            elif object_term_type == RML_BLANK_NODE:
                # raise NotImplementedError("Blank nodes are not implemented, and will not be implemented due to their nature.")
                return None
            elif object_term_type == RML_LITERAL:
                object_map_value = f'"{object_map_value}"'
            return f"?{subject_reference} {predicate} {object_map_value} ."

        

        if object_map_type == RML_REFERENCE:
            object_identifier:str = Identifier.generate_plain_identifier(self.rule, object_map_value)
            object_reference, already_bound = codex.get_id_and_is_bound(object_identifier)
            if object_identifier in encoded_references:
                lines = []
                plain_object_reference, already_bound = codex.get_id_and_is_bound(f"{object_identifier}_plain_{IdGenerator.get_id()}")
                if already_bound:
                    lines.append(f"OPTIONAL{{?{subject_reference} {predicate} ?{plain_object_reference}}}")
                    lines.append(f"OPTIONAL{{BIND(ENCODE_FOR_URI(STR(?{plain_object_reference})) as ?{object_reference}_encoded)}}")
                    lines.append(f"FILTER(!BOUND(?{object_reference}_encoded) || !BOUND(?{plain_object_reference}) || ENCODE_FOR_URI(STR(?{plain_object_reference})) = ?{object_reference}_encoded)")
                else:
                    lines.append(f"OPTIONAL{{?{subject_reference} {predicate} ?{plain_object_reference} . BIND(ENCODE_FOR_URI(STR(?{plain_object_reference})) as ?{object_reference}_encoded)}}")
                return "\n".join(lines)
            else:
                lines = []
                temp_object_reference, already_bound = codex.get_id_and_is_bound(f"{object_identifier}_temp_{IdGenerator.get_id()}")
                if already_bound:
                    lines.append(f"OPTIONAL{{?{subject_reference} {predicate} ?{temp_object_reference}}}")
                    lines.append(f"OPTIONAL{{BIND(?{temp_object_reference} as ?{object_reference})}}")
                    lines.append(f"FILTER(!BOUND(?{object_reference}) || !BOUND(?{temp_object_reference})  || ?{temp_object_reference} = ?{object_reference})")
                else:
                    lines.append(f"OPTIONAL{{?{subject_reference} {predicate} ?{object_reference}}}")
                return "\n".join(lines)
            
        elif object_map_type == RML_TEMPLATE:
            object_identifier:str = Identifier.generate_plain_identifier(self.rule, object_map_value)
            object_reference, already_bound = codex.get_id_and_is_bound(object_identifier)
            lines = []
            lines.append(f"?{subject_reference} {predicate} ?{object_reference}")
            lines.append(f"FILTER(!BOUND(?{object_reference}) || REGEX(STR(?{object_reference}), '{self.rule['object_references_template']}'))")
            evaluated_template = object_references_template
            current_slice = object_reference
            for object in self.rule["object_references"]: 
                current_pre_string = evaluated_template.split("(", 1)[0]
                current_post_string = evaluated_template.split(")", 1)[1]
                next_pre_string = current_post_string.split("(", 1)[0]
                object_identifier = Identifier.generate_plain_identifier(self.rule, object)
                object_reference, already_bound = codex.get_id_and_is_bound(object_identifier)
                next_slice_identifier = f"{object_identifier}_slice_{IdGenerator.get_id()}"
                next_slice = codex.get_id(next_slice_identifier)
                unescaped_current_pre_string = current_pre_string.replace('\\', "")
                unescaped_next_pre_string = next_pre_string.replace('\\', "")
                lines.append(f"{{}} OPTIONAL{{BIND(STRAFTER(STR(?{current_slice}), '{unescaped_current_pre_string}') as ?{next_slice})}}")
                if current_post_string == "":
                    lines.append(f"{{}} OPTIONAL{{BIND(?{next_slice} as ?{object_reference}_encoded)}}") # TODO does this not need the bind/filter flow?
                    lines.append(f"FILTER(!BOUND(?{object_reference}_encoded) || !BOUND(?{next_slice}) || ?{next_slice} = ?{object_reference}_encoded)") # <= untested line
                else:
                    temp_reference_identifier = f"{object_identifier}_temp_{IdGenerator.get_id()}"
                    temp_reference = codex.get_id(temp_reference_identifier)
                    lines.append(
                        f"BIND(STRBEFORE(STR(?{next_slice}), '{unescaped_next_pre_string}') AS ?{temp_reference})"
                    )
                    lines.append(f"{{}} OPTIONAL{{BIND(?{temp_reference} as ?{object_reference}_encoded)}}")
                    lines.append(f"FILTER(!BOUND(?{object_reference}_encoded) || !BOUND(?{temp_reference}) || ?{temp_reference} = ?{object_reference}_encoded)")

                evaluated_template = current_post_string
                current_slice = next_slice
            return "\n".join(lines)

        elif object_map_type == RML_PARENT_TRIPLES_MAP:
            subject_map_value = self.rule["subject_map_value"]
            get_logger().debug(f"Generating parent triples map for {subject_map_value} (subject identifier: {subject_reference})")
            object_parent_triples_map_id = self.rule["object_map_value"]
            object_rule = all_mapping_rules[all_mapping_rules["triples_map_id"] == object_parent_triples_map_id].iloc[0]
            object_map_value = object_rule["subject_map_value"]
            object_reference = codex.get_id(object_map_value)
            get_logger().debug(f"Object map value: {object_map_value} (object reference: {object_reference})")
            predicate = f'<{self.rule["predicate_map_value"]}>'
            get_logger().debug(f"Predicate: {predicate}")
            get_logger().debug(f"Proposed mapping: ?{subject_reference} {predicate} ?{object_reference} (subject map value: {subject_map_value}) (object map value: {object_map_value})")
            return f"?{subject_reference} {predicate} ?{object_reference} ."
            
        else:
            get_logger().error(f"Unsupported object map type: {object_map_type}")
            return None
            
            
class SubjectTriple(QueryTriple):
    def __init__(self, rule: pd.Series):
        super().__init__(rule)

    @property
    def uri_encoded_references(self) -> set[str]:
        return self.subject_references
    
    @property
    def plain_references(self) -> set[str]:
        return set()

    def generate(self, encoded_references: set[str], IdGenerator: IdGenerator, codex: Codex, all_mapping_rules: pd.DataFrame) -> str | None:
        subject_map_value = self.rule["subject_map_value"]
        subject_map_type = self.rule["subject_map_type"]
        subject_term_type = self.rule["subject_termtype"]
        subject_references_template = self.rule["subject_references_template"]
        
        if subject_map_type != RML_TEMPLATE or subject_term_type != RML_IRI:
            get_logger().error(f"Unsupported subject map type: {subject_map_type} or subject term type: {subject_term_type}")
            return None
        
        subject_reference = codex.get_id(subject_map_value)
        
        lines = []
        full_template_reference = subject_reference
        lines.append(f"FILTER(REGEX(STR(?{full_template_reference}), '{self.rule['subject_references_template']}'))")
        evaluated_template = subject_references_template
        current_slice_reference = full_template_reference
        for reference in self.rule["subject_references"]: # we cant use self.object_references here as the order is important (#TODO: refactor self.object_references)
            current_pre_string = evaluated_template.split("(", 1)[0]
            current_post_string = evaluated_template.split(")", 1)[1]
            next_pre_string = current_post_string.split("(", 1)[0]
            reference_identifier = Identifier.generate_plain_identifier(self.rule, reference)
            current_reference, already_bound = codex.get_id_and_is_bound(reference_identifier)
            next_slice_reference_identifier = f"{subject_map_value}_slice_subject_{IdGenerator.get_id()}"
            next_slice_reference = codex.get_id(next_slice_reference_identifier)
            lines.append(f"{{}} OPTIONAL{{BIND(STRAFTER(STR(?{current_slice_reference}), '{current_pre_string}') as ?{next_slice_reference})}}")
            if current_post_string == "":
                lines.append(f"{{}} OPTIONAL{{BIND(?{next_slice_reference} as ?{current_reference}_encoded)}}")
                lines.append(f"FILTER(!BOUND(?{current_reference}_encoded) || !BOUND(?{next_slice_reference}) || ?{next_slice_reference} = ?{current_reference}_encoded)")
            else:
                reference_placeholder = codex.get_id(f"{reference_identifier}_temp_{IdGenerator.get_id()}")
                lines.append(
                    f"BIND(STRBEFORE(STR(?{next_slice_reference}), '{next_pre_string}') AS ?{reference_placeholder})"
                )
                lines.append(f"{{}} OPTIONAL{{BIND(?{reference_placeholder} as ?{current_reference}_encoded)}}")
                lines.append(f"FILTER(!BOUND(?{current_reference}_encoded) || !BOUND(?{reference_placeholder}) || ?{reference_placeholder} = ?{current_reference}_encoded)")
            evaluated_template = current_post_string
            current_slice_reference = next_slice_reference
        return "\n".join(lines)

# endregion

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
        self.codex = Codex()

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

    def generate(self, all_mapping_rules) -> str:
        # select triples using strategy
        inversion_logger.info(f"Selecting triples using {self.selector}")
        selected_triples:list[Triple] = self.selector.select(self.triples)
        subject_count = len(set([triple.rule["subject_map_value"] for triple in selected_triples]))
        triple_count = len(selected_triples)

        all_references = self.references
        uri_encoded_references = self.uri_encoded_references
        plain_references = self.plain_references
        
        if all_references == []:
            inversion_logger.warning("No references found, no query generated")
            return None

        inversion_logger.info(
            f"Selected {triple_count} triples with {subject_count} subjects having:\n\
                {len(uri_encoded_references)} URI encoded references: {uri_encoded_references}\n\
                {len(plain_references)} plain references: {plain_references}\n\
                {len(all_references)} all references: {all_references}")
        triple_strings = []
        
        constant_triples = [triple for triple in selected_triples if triple.rule["object_map_type"] == RML_CONSTANT]
        reference_triples = [triple for triple in selected_triples if triple.rule["object_map_type"] == RML_REFERENCE]
        template_triples = [triple for triple in selected_triples if triple.rule["object_map_type"] == RML_TEMPLATE]
        parent_triples = [triple for triple in selected_triples if triple.rule["object_map_type"] == RML_PARENT_TRIPLES_MAP]
        
        inversion_logger.debug(f"Selected triples: {len(selected_triples)}")
        inversion_logger.debug(f"Constant triples: {len(constant_triples)}")
        inversion_logger.debug(f"Reference triples: {len(reference_triples)}")
        inversion_logger.debug(f"Template triples: {len(template_triples)}")
        inversion_logger.debug(f"Parent triples: {len(parent_triples)}")
                            
        # sorting might improve performance
        for selected_triples in [constant_triples, parent_triples, template_triples, reference_triples]:
            for triple in selected_triples:
                triple_string = triple.generate(uri_encoded_references, self.idGenerator, self.codex, all_mapping_rules)
                if triple_string is not None:
                    triple_strings.append(triple_string)
        
        # for triple in selected_triples:
        #     triple_string = triple.generate(uri_encoded_references, self.idGenerator, self.codex, all_mapping_rules)
        #     if triple_string is not None:
        #         triple_strings.append(triple_string)
                
        pure_references = [f'?{self.codex.get_id(reference)}' for reference in self.pure_references]
        
        select_part = "SELECT " + " ".join(pure_references + [f'?{self.codex.get_id(reference)}_encoded' for reference in uri_encoded_references]) + " WHERE {"
        generated_query = select_part + "\n".join(triple_strings) + "}"
        inversion_logger.debug(self.codex.codex)
        return generated_query.replace("\\", "\\\\")

    def decode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=True)
        for reference in self.uri_encoded_references:
            column_reference = self.codex.get_id(reference)
            column = f"{column_reference}_encoded" 
            df[column] = df[column].apply(url_decode)
            df.rename(columns={column: reference}, inplace=True)
        for reference in self.pure_references:
            column_reference = self.codex.get_id(reference)
            df.rename(columns={column_reference: reference}, inplace=True)
        return df

    def execute_on_endpoint(self, endpoint: Endpoint) -> pd.DataFrame:
        generated_query = self.generate()
        csv_result = endpoint.query(generated_query)
        df = pd.read_csv(StringIO(csv_result))
        return self.decode_dataframe(df)

# region Endpoints
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
        self.delete_after_use = delete_after_use
        with open(url, "r", encoding="utf-8") as f:
            data = f.read()
        self._repoid = hashlib.md5(data.encode("utf-8")).hexdigest()
        inversion_logger.debug(f"Creating repository: {self._repoid}")
        rdf4jconnector = pyrdf4j.rdf4j.RDF4J(rdf4j_base="http://localhost:7200/")
        rdf4jconnector.empty_repository(self._repoid)
        
        # Create GraphDB repository configuration
        config = f"""
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>.
        @prefix rep: <http://www.openrdf.org/config/repository#>.
        @prefix sr: <http://www.openrdf.org/config/repository/sail#>.
        @prefix sail: <http://www.openrdf.org/config/sail#>.
        @prefix graphdb: <http://www.ontotext.com/config/graphdb#>.

        [] a rep:Repository ;
            rep:repositoryID "{self._repoid}" ;
            rdfs:label "GraphDB Repository" ;
            rep:repositoryImpl [
                rep:repositoryType "graphdb:SailRepository" ;
                sr:sailImpl [
                    sail:sailType "graphdb:Sail" ;
                    graphdb:entity-id-size  "32" ;
                ]
            ].
        """
        
        rdf4jconnector.create_repository(self._repoid, config=config, accept_existing=True)
        try:
            rdf4jconnector.add_data_to_repo(self._repoid, data, "text/x-nquads")
        except pyrdf4j.errors.TerminatingError as e:
            logging.error("Invalid RDF data")
        time.sleep(1)
        self._sparql = SPARQLWrapper(
            f"http://localhost:7200/repositories/{self._repoid}"
        )
        self._sparql.setReturnFormat(CSV)
        self._sparql.setMethod("POST")
        self._sparql.setRequestMethod("postdirectly")

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

# endregion


class JSONPathFunctions:
    def __init__(self):
        """Do not instantiate this class

        Raises:
            NotImplementedError: This class is should not be instantiated
        """        
        raise NotImplementedError("This class is should not be instantiated")
    
    @staticmethod
    def list_path_steps(jsonpath: jsonpath_ng.JSONPath) -> list[jsonpath_ng.JSONPath]:
        steps = []
        current = jsonpath
        while isinstance(current, jsonpath_ng.Child):
            steps.append(current.right)
            current = current.left
        steps.append(current)
        return steps[::-1]
    
    def find_top(self, jsonpath: jsonpath_ng.JSONPath) -> jsonpath_ng.JSONPath:
        return self.list_path_steps(jsonpath)[0]
    
    def get_json_path(steps: list[Node]) -> jsonpath_ng.JSONPath:
        if len(steps) == 0:
            return None
        if isinstance(steps[0], Root):
            current = jsonpath_ng.Root()
        if isinstance(steps[0], Object):
            current = jsonpath_ng.Fields(steps[0].values[0])
        if isinstance(steps[0], Array):
            current = jsonpath_ng.Slice()
        for step in steps[1:]:
            if isinstance(step, Object):
                current = jsonpath_ng.Child(current, jsonpath_ng.Fields(step.values[0]))
            if isinstance(step, Array):
                current = jsonpath_ng.Child(current, jsonpath_ng.Slice())
        return current
    
    @staticmethod
    @functools.cache
    def normalize_json_path(path: str) -> str:
        parsed:jsonpath_ng.Child = jsonpath_ng.parse(path)
        return str(parsed)
    
    @staticmethod
    def extend_string_path(path: str, extension: str) -> str:
        """Extends a string JSON path

        Args:
            path (str): base path
            extension (str): extension to add

        Returns:
            str: extended path
        """
        if ' ' in extension:
            new_path = f"{path}['{extension}']"
        else:
            new_path = f"{path}.{extension}"
        return JSONPathFunctions.normalize_json_path(new_path)

class CSVTemplate(Template):
    def __init__(self):
        pass
    
    def create_template(self) -> str:
        return "No real CSV template is created as we can just dump the dataframe to csv"
    
    def fill_data(self, data: pd.DataFrame, source_name: str) -> str:
        return data.to_csv(index=False)
    
    @property
    def columns_decoded(self) -> bool:
        return True

class JSONTemplate(Template): # TODO: cleanup (split non-class dependent functions to somewhere else)
    """Template for JSON data, filling the template will either be done by passing the data to the nodes, or by simply using string templates
    Passing the data would be more robust, but come at a performance cost
    String templates could lead to unforeseen issues, but would be faster (probably)... which could count in huge datasets
    """
    def __init__(self):
        self.paths:list[jsonpath_ng.JSONPath] = []
    
    @property
    def columns_decoded(self) -> bool:
        return False
    
    @property
    def root(self) -> Node:
        """Create the root from the paths
        
        Returns:
            Node: The root node
        """
        if len(self.paths) == 0:
            return Root()
        # first try to find a path connected to the root
        root_path = None
        for path in self.paths:
            top_steps = JSONPathFunctions.list_path_steps(path)
            if isinstance(top_steps[0], jsonpath_ng.Root):
                root_path = path
                break
        if root_path is None:
            # will be implemented later, probably... or a more descriptive error will be raised
            raise ValueError("No root path found")
        root = self.create_node_tree(JSONPathFunctions.list_path_steps(root_path))
        
        for path in self.paths:
            # merge the paths into the tree
            node = self.create_node_tree(JSONPathFunctions.list_path_steps(path))
            self.merge_node_trees(root, node)
        return root
    
    def add_path(self, jsonpath: jsonpath_ng.JSONPath|str) -> bool:
        """Add a full path to the template
        Args:
            jsonpath (jsonpath_ng.JSONPath or str): The path to add eg. $.students[*].name
        
        Returns:
            True if the path was added, False if the path was already present
        """    
        if isinstance(jsonpath, str):
            jsonpath = jsonpath_ng.parse(jsonpath)
        if jsonpath in self.paths:
            return False
        self.paths.append(jsonpath)
        return True
    
    def create_node_tree(self, steps: list[jsonpath_ng.JSONPath]) -> Node:
        """Create a tree of nodes from the steps in the path
        
        Args:
            steps (list[jsonpath_ng.JSONPath]): The steps in the path
        
        Returns:
            Node: The root node of the tree
        """    
        if len(steps) == 0:
            return None
        if len(steps) == 1:
            return Object(values=[steps[0].fields[0]])
        root_step = steps[0]
        if isinstance(root_step, jsonpath_ng.Root):
            root = Root()
        if isinstance(root_step, jsonpath_ng.Fields):
            root = Object()
            key = root_step.fields[0]
        if isinstance(root_step, jsonpath_ng.Slice):
            root = Array()
        current = root
        for step in steps[1:-1]:
            if isinstance(step, jsonpath_ng.Fields):
                next = Object()
                key = step.fields[0]
            if isinstance(step, jsonpath_ng.Slice):
                next = Array()
            if isinstance(current, Object):
                current.add_child(key, next)
            elif isinstance(current, Array):
                current.content = next
            elif isinstance(current, Root):
                current.child = next
            current = next
        leaf = Object(values=[steps[-1].fields[0]])
        if isinstance(current, Object):
            current.add_child(key, leaf)
        elif isinstance(current, Array):
            current.content = leaf
        return root
    
    def merge_node_trees(self, base: Node, other: Node):
        """Merge two node trees
        
        Args:
            base (Node): The base tree
            other (Node): The other tree
        """
        if isinstance(base, Object):
            if isinstance(other, Object):
                for key, child in other.children.items():
                    if key in base.children.keys():
                        self.merge_node_trees(base.children[key], child)
                    else:
                        base.children[key] = child
                for value in other.values:
                    if value not in base.values:
                        base.values.append(value)
            else:
                raise ValueError("Cannot merge Object with non-Object")
        if isinstance(base, Array):
            if isinstance(other, Array):
                self.merge_node_trees(base.content, other.content)
            else:
                raise ValueError("Cannot merge Array with non-Array")
        if isinstance(base, Root):
            if isinstance(other, Root):
                self.merge_node_trees(base.child, other.child)
            else:
                raise ValueError("Cannot merge Root with non-Root")
                
    def create_template(self) -> str:
        """Create a template from the paths, this is only for demonstration purposes as filling the data is more complex
        
        Returns:
            str: The template
        """           
        return self.root.to_template()
    
    def fill_data(self, data: pd.DataFrame, source_name: str) -> str:
        """Fill the template with data
        
        Args:
            data (pd.DataFrame): The data to fill the template with
            
        Returns:
            str: The filled template
        """
        return self.root.fill(data)
    
    def __str__(self):
        return f"JSONTemplate({self.paths})"

class RDBTemplate(Template):
    def __init__(self):
        pass

    def create_template(self) -> str:
        return "RDB template: structure will be determined by the database schema"

    def fill_data(self, data: pd.DataFrame, table_name: str) -> str:
        create_table_sql = self.generate_create_table_sql(table_name, data)
        insert_data_sql = self.generate_insert_data_sql(table_name, data)
        return create_table_sql + "\n\n" + insert_data_sql

    def generate_create_table_sql(self, table_name: str, df: pd.DataFrame) -> str:
        columns = [f"{col} TEXT" for col in df.columns]  # Assuming all columns are TEXT for simplicity
        columns_str = ", ".join(columns)
        return f"CREATE TABLE {table_name} ({columns_str});"

    def generate_insert_data_sql(self, table_name: str, df: pd.DataFrame) -> str:
        insert_statements = []
        for _, row in df.iterrows():
            values = [f"'{str(value).replace('', '')}'" for value in row]
            values_str = ", ".join(values)
            insert_statements.append(f"INSERT INTO {table_name} VALUES ({values_str});")
        return "\n".join(insert_statements)

    @property
    def columns_decoded(self) -> bool:
        return True

    @property
    def columns_decoded(self) -> bool:
        return True

class Object(Node):
    def __init__(self, children: dict[str, Node] = None, values: list[str] = None):
        self._parent_path = ""
        if children is None:
            children = {}
        if values is None:
            values = []
        self.children = children
        self.values = values
    
    def add_child(self, key: str, child: Node):
        self.children[key] = child
        child.parent_path = self.path + "." + key
    
    @property
    def path(self) -> str:
        return self.parent_path 
    
    @property
    def parent_path(self) -> str:
        return self._parent_path
    
    @parent_path.setter
    def parent_path(self, value: str):
        self._parent_path = value
    
    def find(self, key: str) -> Node|None:
        if key in self.children.keys():
            return self.children[key]
        for child in self.children.values():
            if child.find(key) is not None:
                return child
            
    def to_template(self) -> str:
        child_strings = [f'"{key}": {child.to_template()}' for key, child in self.children.items()]
        value_strings = [f'"{value}": "${value}"' for value in self.values]
        return "{" + \
            ", ".join(child_strings + value_strings) + \
            "}"
            
    def fill(self, data: pd.DataFrame) -> str:
        """Fill the template with data,
        automatically groups the data into slices, for each array in the data

        Args:
            data (pd.DataFrame): The data to fill the template with

        Returns:
            str: The filled template
        """
        paths = [JSONPathFunctions.normalize_json_path(f"{self.path}.['{value}']") for value in self.values]
        filled_values = [f'"{value}": "{data[path].iloc[0]}"' for value, path in zip(self.values, paths)]
        filled_children = [f'"{key}": {child.fill(data)}' for key, child in self.children.items()]
        return "{" + ", ".join(filled_values + filled_children) + "}"
    
    def get_slice_columns(self) -> list[str]:
        columns = [JSONPathFunctions.normalize_json_path(f"{self.path}.['{value}']") for value in self.values]
        for child in self.children.values():
            if isinstance(child, Object):
                columns.extend(child.get_slice_columns())
        return columns
        
class Array(Node):
    def __init__(self, content: Node = None):
        self._parent_path = ""
        self._content = None
        if content is not None:
            self.content = content
    
    @property
    def path(self) -> str:
        return self.parent_path + "[*]"
    
    @property
    def parent_path(self) -> str:
        return self._parent_path
    
    @parent_path.setter
    def parent_path(self, value: str):
        self._parent_path = value
    
    @property
    def content(self) -> Node:
        return self._content
    
    @content.setter
    def content(self, value: Node):
        self._content = value
        self._content.parent_path = self.path

    def to_template(self) -> str:
        return "[" + self.content.to_template() + "]"
    
    def fill(self, data: pd.DataFrame) -> str:
        if isinstance(self.content, Object):
            columns = self.content.get_slice_columns()
            
            grouped_data = data.groupby(columns, dropna=False)
            content_lines = []
            
            for _, group in grouped_data:
                if len(group) == 0:
                    continue
                content_lines.append(self.content.fill(group))
            
            joined_content = ", ".join(content_lines)
            return "[" + joined_content + "]"
        
        return "[" + self.content.fill(data) + "]"



class Root(Node):
    def __init__(self, child: Node = None):
        self._child = None
        if child is not None:
            self.child = child
            
    @property
    def child(self) -> Node:
        return self._child
    
    @child.setter
    def child(self, value: Node):
        self._child = value
        self._child.parent_path = "$"
        
    @property
    def path(self) -> str:
        return "$"
    
    @property
    def parent_path(self) -> str:
        return ""
        
    def find(self, key: str) -> Node|None:
        if self.child is None:
            return None
        return self.child.find(key)
    
    def to_template(self) -> str:
        if self.child is None:
            return "{}"
        return self.child.to_template()
    
    def fill(self, data: pd.DataFrame) -> str:
        if self.child is None:
            return "{}"
        return self.child.fill(data)
        

inversion_logger = logging.getLogger("inversion")

def get_logger() -> logging.Logger:
    return inversion_logger

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

def retrieve_data(
    mapping_rules: pd.DataFrame,
    source_rules: pd.DataFrame,
    endpoint: Endpoint,
    decode_columns: bool = False,
    try_cached: bool = False
) -> pd.DataFrame | None:
    retrieve_data_start_time = time.time()
    source = source_rules.iloc[0]['logical_source_value']
    inversion_logger.debug(f"Processing source {source}")

    # Generate a hash of the source to use as a filename
    source_hash = hashlib.md5(source.encode()).hexdigest()
    filename = f"{source_hash}_table.csv"

    triples: list[QueryTriple] = [
        QueryTriple(rule) for _, rule in source_rules.iterrows() if rule["object_map_type"] not in [RML_BLANK_NODE]
    ]
    triples.extend(
        SubjectTriple(subject_rules.iloc[0])
        for subject, subject_rules in source_rules.groupby("subject_map_value", dropna=False)
    )

    query = Query(triples)
    generated_query = query.generate(mapping_rules)

    if generated_query is None:
        inversion_logger.warning("No query generated (no references found)")
        return None

    inversion_logger.debug(generated_query)

    try:
        if try_cached and pathlib.Path(filename).exists():
            inversion_logger.info(f"Using cached data for {source}")
            end_query_time = time.time()
        else:
            start_query_time = time.time()
            inversion_logger.debug(f"Time to generate query: {start_query_time - retrieve_data_start_time}s")
            result = endpoint.query(generated_query)
            end_query_time = time.time()
            inversion_logger.debug(f"Time to query endpoint: {end_query_time - start_query_time}s")

            with open(filename, "w", encoding="utf-8") as file:
                file.write(result)

        df = pd.read_csv(filename, dtype=str)

        if decode_columns:
            df = query.decode_dataframe(df)

        convert_time = time.time()
        inversion_logger.debug(f"Time to convert data: {convert_time - end_query_time}s")
        return df

    except Exception as e:
        inversion_logger.warning(f"Error while querying endpoint: {e}")
        raise

def generate_template(source_rules: pd.DataFrame) -> Template | None:
    source_type = source_rules.iloc[0]["source_type"]
    inversion_logger.info(f"Generating template for source type {source_type}")

    if source_type == "JSON":
        template = JSONTemplate()
        for _, rule in source_rules.iterrows():
            if rule["object_map_type"] in [RML_BLANK_NODE, RML_PARENT_TRIPLES_MAP]:
                continue
            iterator = rule["iterator"]
            for value in rule["subject_references"] + rule["predicate_references"] + rule["object_references"]:
                splitted = value.split(".")
                predecessors = '.'.join(splitted[:-1])
                path = f"{iterator}.{predecessors}['{splitted[-1]}']"
                template.add_path(path)
        inversion_logger.debug(json.dumps(json.loads(template.create_template()), indent=4))
        return template
    elif source_type == "CSV":
        return CSVTemplate()
    elif source_type == "RDB":
        return RDBTemplate()
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

def test_logging_setup(testID: str):
    if not os.path.exists(TEST_LOG_FOLDER):
        os.mkdir(TEST_LOG_FOLDER)

    if os.path.exists(TEST_LOG_FOLDER / f"{testID}.log"):
        os.remove(TEST_LOG_FOLDER / f"{testID}.log")
    inversion_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    file_logger = logging.FileHandler(TEST_LOG_FOLDER / f"{testID}.log")
    file_logger.setLevel(logging.DEBUG)
    file_logger.setFormatter(formatter)
    inversion_logger.addHandler(file_logger)
    inversion_logger.setLevel(logging.DEBUG)
    
def inversion(config_file: str | pathlib.Path, testID: str = None) -> dict[str, str]:
    results = {}
    start_time = time.time()
    if testID is not None:
        test_logging_setup(testID)  
    config = load_config_from_argument(config_file)
    mappings: pd.DataFrame
    try:
        mappings, _ = retrieve_mappings(config)
    except ValueError as e:
        if str(e) == "Not supported query type!":
            inversion_logger.warning(f"Invalid SQL query in mapping")
        return results
    except KeyError as e:
        if str(e) == "'object_map'":
            inversion_logger.warning(f"Mapping with missing information. Skipping.")
        return results
    try:
        endpoint = EndpointFactory.create(config)
    except FileNotFoundError:
        inversion_logger.warning(f"Output file not found. Skipping inversion.")
        return results
    insert_columns(mappings)
    setup_done_time = time.time()
    inversion_logger.debug(f"Starting sources generation, {setup_done_time - start_time}s used for setup")
    for source, source_rules in mappings.groupby("logical_source_value"):
        inversion_logger.info(f"Processing source {source}")
        template_generation_start_time = time.time()
        template = generate_template(source_rules)
        data_retrieval_start_time = time.time()
        inversion_logger.debug(f"Starting data retrieval, {data_retrieval_start_time - template_generation_start_time}s used for template generation")
        source_data = retrieve_data(mappings, source_rules, endpoint, decode_columns=True)
        if source_data is None:
            results[source] = ""
            inversion_logger.warning(f"No data generated for {source}")
            continue
        try:
            template_filling_start_time = time.time()
            inversion_logger.debug(f"Starting template filling, {template_filling_start_time - data_retrieval_start_time}s used for data retrieval")
            filled_source = template.fill_data(source_data, source)
            results[source] = filled_source
        except AttributeError as e:
            inversion_logger.error(f"Error while filling template: {e}")
            raise e
        source_end_time = time.time()
        inversion_logger.info(f"Source filled in {source_end_time - template_filling_start_time}s")
        inversion_logger.info(f"Source {source} processed in {source_end_time - template_generation_start_time}s")
    return results
    
def url_decode(url):
    try:
        # check if url is a string
        return unquote(url) if isinstance(url, str) else url
    except Exception as e:
        # Handle invalid URLs or other decoding errors
        return url

def logging_setup():
    if os.path.exists("inversion.log"):
        try:
            os.remove("inversion.log")
        except Exception as e:
            print(f"Error while removing inversion.log: {e}")
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

def main():
    logging_setup()
    # ignore morph_kgc FutureWarning logs
    warnings.simplefilter(action="ignore", category=FutureWarning)
    test2()


def test():
    this_file_path = pathlib.Path(__file__).resolve()
    implementation_dir = this_file_path.parent
    metadata_path = implementation_dir / "rml-test-cases" / "metadata.csv"
    testcases_path = implementation_dir / "rml-test-cases" / "test-cases"
    with open(metadata_path, "r") as file:
        tests_df: pd.DataFrame = pd.read_csv(file)
    tests_with_output = tests_df[tests_df["error expected?"] == False]
    
    selected_tests_ids = ["15a"]
    
    selected_tests = tests_with_output[tests_with_output["better RML id"].isin(selected_tests_ids)]
    
    for _, row in selected_tests.iterrows():
        inversion_logger.info(f'Running test {row["RML id"]}, ({row["better RML id"]})')
        os.chdir(testcases_path / row["RML id"])
        inversion(MORPH_CONFIG, testID=row["RML id"])

def small_test():
    test_folder_path = "C:\Github\Knowledge-graphs-inversion\Implementation\Tests\Inversion_tests\Temp"
    os.chdir(test_folder_path)
    inversion(MORPH_CONFIG, testID="test")
    
def rml_test_cases():
    bad_tests = ["4a", "16a", "18a", "20a", "21a", "22a", "23a", "24a", "26a", "27a", "28a", "31a", "36a", "37a", "40a", "41a", "42a", "56a", "57a", "58a", "59a"]
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

def test2():
    os.chdir(r"C:\Github\Knowledge-graphs-inversion\Implementation\Tests\Inversion_tests\Temp")
    config = load_config_from_argument(MORPH_CONFIG)
    mappings: pd.DataFrame
    mappings, _ = retrieve_mappings(config)
    insert_columns(mappings)
    results = {}
    for source, source_rules in mappings.groupby("logical_source_value"):
        inversion_logger.info(f"Processing source {source}")
        template = generate_template(source_rules)
        # for _, mapping in source_rules.iterrows():
        #     get_logger().debug(mapping['object_join_conditions'])
        # for _, mapping in source_rules.iterrows():
        #     get_logger().debug(mapping)
        data = retrieve_data(mappings, source_rules, EndpointFactory.create(config), decode_columns=True)
        get_logger().debug(template.fill_data(data))

if __name__ == "__main__":
    main()
