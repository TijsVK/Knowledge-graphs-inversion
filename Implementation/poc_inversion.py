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
from typing import Self  

# region Constants

QUERY_MINIMAL = 0
QUERY_REDUCED = 1
QUERY_SIMPLE = 2
QUERY_FULL = 3

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

class Selector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select(self):
        pass
    
class Node(ABC):
    def find(self, key: str) -> Self|None:
        raise NotImplementedError
    
    @abstractmethod
    def to_template(self) -> str:
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

class Hexer:
    @staticmethod
    def encode(x: str) -> str:
        return x.encode("utf-8").hex()
    
    @staticmethod
    def decode(x: str) -> str:
        return bytes.fromhex(x).decode("utf-8")
    
    @staticmethod
    def shift_10(x: str) -> str:
        """shifts the hex string by 10, eg 0123456789abcdef becomes abcdefghijklmnop

        Args:
            x (str): _description_

        Returns:
            str: _description_
        """
        hex_numbers = "0123456789abcdef"
        converted = "abcdefghijklmnop"
        return x.translate(str.maketrans(hex_numbers, converted))
    
    @staticmethod
    def unshift_10(x: str) -> str:
        """unshifts the hex string by 10, eg abcdefghijklmnop becomes 0123456789abcdef

        Args:
            x (str): _description_

        Returns:
            str: _description_
        """
        hex_numbers = "0123456789abcdef"
        converted = "abcdefghijklmnop"
        return x.translate(str.maketrans(converted, hex_numbers))
        

# endregion

# region Triples

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
        object_references_template = self.rule["object_references_template"]
        
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
            evaluated_template = object_references_template
            current_reference = full_template_reference
            for reference in self.rule["object_references"]: # we cant use self.object_references here as the order is important (#TODO: refactor self.object_references)
                current_pre_string = evaluated_template.split("(", 1)[0]
                current_post_string = evaluated_template.split(")", 1)[1]
                next_pre_string = current_post_string.split("(", 1)[0]
                reference_byte_string = reference.encode("utf-8")
                reference_hex = reference_byte_string.hex()
                next_reference = f"{object_reference_hex}_slice_{IdGenerator.get_id()}"
                unescaped_current_pre_string = current_pre_string.replace('\\', "")
                unescaped_next_pre_string = next_pre_string.replace('\\', "")
                lines.append(f"{{}} OPTIONAL{{BIND(STRAFTER(STR(?{current_reference}), '{unescaped_current_pre_string}') as ?{next_reference})}}")
                if current_post_string == "":
                    lines.append(f"{{}} OPTIONAL{{BIND(?{next_reference} as ?{reference_hex}_encoded)}}")
                else:
                    reference_placeholder = f"{reference_hex}_{IdGenerator.get_id()}"
                    lines.append(
                        f"BIND(STRBEFORE(STR(?{next_reference}), '{unescaped_next_pre_string}') AS ?{reference_placeholder})"
                    )
                    lines.append(f"{{}} OPTIONAL{{BIND(?{reference_placeholder} as ?{reference_hex}_encoded)}}")
                    lines.append(f"FILTER(!BOUND(?{reference_hex}_encoded) || ?{reference_placeholder} = ?{reference_hex}_encoded)")

                evaluated_template = current_post_string
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
        subject_term_type = self.rule["subject_termtype"]
        subject_references_template = self.rule["subject_references_template"]
        
        if subject_map_type != RML_TEMPLATE or subject_term_type != RML_IRI:
            return None
        
        subject_reference_byte_string = subject_map_value.encode("utf-8")
        subject_reference_hex = subject_reference_byte_string.hex()
        
        lines = []
        full_template_reference = f"{subject_reference_hex}"
        lines.append(f"FILTER(REGEX(STR(?{full_template_reference}), '{self.rule['subject_references_template']}'))")
        evaluated_template = subject_references_template
        current_reference = full_template_reference
        for reference in self.rule["subject_references"]: # we cant use self.object_references here as the order is important (#TODO: refactor self.object_references)
            current_pre_string = evaluated_template.split("(", 1)[0]
            current_post_string = evaluated_template.split(")", 1)[1]
            next_pre_string = current_post_string.split("(", 1)[0]
            reference_byte_string = reference.encode("utf-8")
            reference_hex = reference_byte_string.hex()
            next_reference = f"{subject_reference_hex}_slice_subject_{IdGenerator.get_id()}"
            lines.append(f"{{}} OPTIONAL{{BIND(STRAFTER(STR(?{current_reference}), '{current_pre_string}') as ?{next_reference})}}")
            if current_post_string == "":
                lines.append(f"{{}} OPTIONAL{{BIND(?{next_reference} as ?{reference_hex}_encoded)}}")
            else:
                reference_placeholder = f"{reference_hex}_{IdGenerator.get_id()}"
                lines.append(
                    f"BIND(STRBEFORE(STR(?{next_reference}), '{next_pre_string}') AS ?{reference_placeholder})"
                )
                lines.append(f"{{}} OPTIONAL{{BIND(?{reference_placeholder} as ?{reference_hex}_encoded)}}")
                lines.append(f"FILTER(!BOUND(?{reference_hex}_encoded) || ?{reference_placeholder} = ?{reference_hex}_encoded)")

            evaluated_template = current_post_string
            current_reference = next_reference
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

class JSONTemplate:
    """Template for JSON data, filling the template will either be done by passing the data to the nodes, or by simply using string templates
    Passing the data would be more robust, but come at a performance cost
    String templates could lead to unforeseen issues, but would be faster (probably)... which could count in huge datasets
    """
    def __init__(self):
        self.paths:list[jsonpath_ng.JSONPath] = []
    
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
            return Leaf(steps[0])
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
                current.children[key] = next
            elif isinstance(current, Array):
                current.content.append(next)
            elif isinstance(current, Root):
                current.child = next
            current = next
        leaf = Leaf([steps[-1].fields[0]])
        if isinstance(current, Object):
            current.children[key] = leaf
        elif isinstance(current, Array):
            current.content.append(leaf)
        return root
                
    
    def create_template(self) -> str:
        """Create a template from the paths, later probably need to make a TemplateBuilder class to handle various options in generation for unknown (wildcard) paths
        
        Returns:
            str: The template
        """    
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
        node_tree = self.create_node_tree(JSONPathFunctions.list_path_steps(root_path))
    
    def __str__(self):
        return f"JSONTemplate({self.paths})"

            
# class Branch(Node):
#     def __init__(self, named_children: dict[str, Node] = None, unnamed_children: list[Node] = None):
#         if named_children is None:
#             named_children = {}
#         if unnamed_children is None:
#             unnamed_children = []
#         self.named_children = named_children
#         self.unnamed_children = unnamed_children

class Object(Node):
    def __init__(self, children: dict[str, Node] = None):
        if children is None:
            children = {}
        self.children = children
    
    def find(self, key: str) -> Node|None:
        if key in self.children.keys():
            return self.children[key]
        for child in self.children.values():
            if child.find(key) is not None:
                return child
            
    def to_template(self, human_readable: bool = False) -> str:
        return "{" + ", ".join([f'"{key}": {child.to_template(human_readable)}' for key, child in self.children.items()]) + "}"
        
class Array(Node):
    def __init__(self, content: list[Node] = None):
        if content is None:
            content = []
        self.content = content
        
    def find(self, key: str) -> Node|None:
        for child in self.content:
            if child.find(key) is not None:
                return child
            
    def to_template(self, human_readable: bool = False) -> str:
        return "[" + ", ".join([child.to_template(human_readable) for child in self.content]) + "]"

class Root(Node):
    def __init__(self, child: Node = None):
        if child is None:
            child = None
        
    def find(self, key: str) -> Node|None:
        return self.child.find(key)
    
    def to_template(self, human_readable: bool = False) -> str:
        return self.child.to_template(human_readable)

class Leaf(Node):
    def __init__(self, values: list[str] = None):
        if values is None:
            values = []
        self.values = values
        
    def __str__(self) -> str:
        return f"Leaf({self.values})"
    
    def to_template(self, human_readable: bool = False) -> str:
        """A leaf is a dictionary with a key and value for each item, eg. {"name": $name} (encoding should be done later)

        Returns:
            str: The template
        """
        return "{" + ", ".join([f'"{value}": ${value}' for value in self.values]) + "}"
        

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
    if len(iterator_result) == 0:
        return None
    else:
        return list(iterator_result.values())[0] # for now, just return the first iterators result

def retrieve_data_from_full_source(
        mapping_rules: pd.DataFrame, source_rules: pd.DataFrame, endpoint: Endpoint
) -> pd.DataFrame | None:
    inversion_logger.debug(f"Processing source {source_rules.iloc[0]['logical_source_value']}")
    for _, rule in source_rules.iterrows():
        for key, value in rule.items():
            inversion_logger.debug(f"{key}: {value}")
    triples: list[QueryTriple] = [
        QueryTriple(rule) for _, rule in source_rules.iterrows()
    ]
    triples.extend(
        SubjectTriple(subject_rules.iloc[0])
        for subject, subject_rules in source_rules.groupby(
            "subject_map_value", dropna=False
        )
    )
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
        except Exception as e:
            inversion_logger.warning(f"Error while querying endpoint: {e}")
            return None
    return decoded_df 

def generate_template(source_rules: pd.DataFrame) -> str:
    source_type = source_rules.iloc[0]["source_type"]
    inversion_logger.info(f"Generating template for source type {source_type}")

    if source_type == "JSON":
        template = JSONTemplate()
        for _, rule in source_rules.iterrows():
            iterator = rule["iterator"]
            # jsonpath = iterator + '.' + (Hexer.encode(value) for values in rule["subject_map_value"])
            for value in rule["subject_references"] + rule["predicate_references"] + rule["object_references"]:
                path = f"{iterator}.['{value}']"
                template.add_path(path)
        template.create_template()
        return template

def test_logging_setup(testID: str):
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
    if testID is not None:
        test_logging_setup(testID)
    config = load_config_from_argument(config_file)
    mappings: pd.DataFrame
    mappings, _ = retrieve_mappings(config)
    endpoint = EndpointFactory.create(config)
    insert_columns(mappings)
    results = {}
    for source, source_rules in mappings.groupby("logical_source_value"):
        inversion_logger.info(f"Processing source {source}")
        template = generate_template(source_rules)
        source_data = retrieve_data_from_full_source(mappings, source_rules, endpoint)
        if source_data is None:
            results[source] = ""
            inversion_logger.warning(f"No data generated for {source}")
            continue
        if source_rules.iloc[0]["source_type"] == "CSV":
            results[source] = source_data.to_csv(index=False)
        else:
            results[source] = ""
            inversion_logger.warning(f"Source type {source_rules.iloc[0]['source_type']} not supported yet")
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


def test():
    logging_setup()
    warnings.simplefilter(action="ignore", category=FutureWarning)
    this_file_path = pathlib.Path(__file__).resolve()
    implementation_dir = this_file_path.parent
    metadata_path = implementation_dir / "rml-test-cases" / "metadata.csv"
    testcases_path = implementation_dir / "rml-test-cases" / "test-cases"
    with open(metadata_path, "r") as file:
        tests_df: pd.DataFrame = pd.read_csv(file)
    tests_with_output = tests_df[tests_df["error expected?"] == False]
    # only CSV tests for now
    tests_with_output = tests_with_output[tests_with_output["data format"] == "JSON"]
    
    selected_tests_ids = ["36b", "40b"]
    
    selected_tests = tests_with_output[tests_with_output["better RML id"].isin(selected_tests_ids)]
    
    for _, row in selected_tests.iterrows():
        inversion_logger.info(f'Running test {row["RML id"]}, ({row["better RML id"]})')
        os.chdir(testcases_path / row["RML id"])
        config = load_config_from_argument(MORPH_CONFIG)
        mappings: pd.DataFrame
        mappings, _ = retrieve_mappings(config)
        insert_columns(mappings)
        results = {}
        for source, source_rules in mappings.groupby("logical_source_value"):
            template = generate_template(source_rules)
    
    expr:jsonpath_ng.Child = jsonpath_ng.parse("$.teachers[*].Name")
    print(expr.__str__())
    print(expr.__repr__())
    students_json_string = """{
        "teachers": [{
                "Country Code": 1,
                "Name":"Jupiter"
            },
            {
                "Country Code": 2,
                "Name":"Saturn"
            },
            {
                "Country Code": 3,
                "Name":"Uranus"
            }
        ]
    }"""
    rights = []
    current = expr
    while isinstance(current, jsonpath_ng.Child):
        rights.append(current.right)
        current = current.left
    rights.append(current)
    inverted_rights = rights[::-1]
    print(inverted_rights)
    students_json = json.loads(students_json_string)
    print([match.value for match in expr.find(students_json)])
    return

def small_test():
    logging_setup()
    warnings.simplefilter(action="ignore", category=FutureWarning)
    template = JSONTemplate()
    template.add_path("$.students[*].card.name")
    template.add_path("$.students[*].card.telephone_number")
    template.add_path("$.students[*].card.school")
    template.create_template()

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

if __name__ == "__main__":
    logging_setup()
    warnings.simplefilter(action="ignore", category=FutureWarning)
    small_test()
