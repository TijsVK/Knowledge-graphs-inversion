import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, Any
from poc_inversion import Endpoint, Query, QueryTriple, SubjectTriple, insert_columns, generate_template


class DatabaseEndpoint(Endpoint):
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)

    def query(self, query: str) -> str:
        with self.engine.connect() as connection:
            result = connection.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df.to_csv(index=False)

def inversion_db(mapping_rules: pd.DataFrame, actual_output: str, connection_string: str) -> Dict[str, Any]:
    endpoint = DatabaseEndpoint(connection_string)
    new_mapping_rules = insert_columns(mapping_rules)
    results = {}

    for source, source_rules in new_mapping_rules.groupby("logical_source_value"):
        template = generate_template(source_rules)
        source_data = retrieve_data(mapping_rules, source_rules, endpoint, decode_columns=True)

        if source_data is None:
            results[source] = ""
            continue

        try:
            filled_source = template.fill_data(source_data)
            results[source] = filled_source
        except AttributeError as e:
            print(f"Error while filling template: {e}")
            raise e

    return results

def retrieve_data(
    mapping_rules: pd.DataFrame, source_rules: pd.DataFrame, endpoint: DatabaseEndpoint, decode_columns: bool = False
) -> pd.DataFrame | None:
    triples: list[QueryTriple] = [
        QueryTriple(rule) for _, rule in source_rules.iterrows() if rule["object_map_type"] != "http://w3id.org/rml/BlankNode"
    ]
    triples.extend(
        SubjectTriple(subject_rules.iloc[0])
        for subject, subject_rules in source_rules.groupby(
            "subject_map_value", dropna=False
        )
    )
    query = Query(triples)
    generated_query = query.generate(mapping_rules)

    if generated_query is None:
        print("No query generated (no references found)")
        return None
    else:
        try:
            result = endpoint.query(generated_query)
            df = pd.read_csv(pd.compat.StringIO(result), dtype=str)
            if decode_columns:
                df = query.decode_dataframe(df)
            return df
        except Exception as e:
            print(f"Error while querying endpoint: {e}")
            raise

def compare_database_structures(original_structure: Dict[str, Any], inverted_structure: Dict[str, Any]) -> bool:
    # Implement comparison logic here
    # This is a placeholder implementation
    return original_structure == inverted_structure

def get_database_structure(connection_string: str) -> Dict[str, Any]:
    engine = create_engine(connection_string)
    with engine.connect() as connection:
        # Get table names
        table_query = "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';"
        tables = pd.read_sql(table_query, connection)
        
        structure = {}
        for table in tables['tablename']:
            # Get column information for each table
            column_query = f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table}';
            """
            columns = pd.read_sql(column_query, connection)
            structure[table] = dict(zip(columns['column_name'], columns['data_type']))
    
    return structure

def create_sql_from_csv(csv_data: str, table_name: str) -> str:
    df = pd.read_csv(pd.compat.StringIO(csv_data))
    create_table_sql = f"CREATE TABLE {table_name} (\n"
    create_table_sql += ",\n".join([f"    {col} TEXT" for col in df.columns])
    create_table_sql += "\n);"
    
    insert_sql = f"INSERT INTO {table_name} VALUES\n"
    insert_sql += ",\n".join([
        "    (" + ", ".join([f"'{str(value).replace('', '')}'" for value in row]) + ")"
        for _, row in df.iterrows()
    ])
    insert_sql += ";"
    
    return create_table_sql + "\n\n" + insert_sql
    
def invert_and_compare(actual_output: str, mapping_rules: pd.DataFrame, connection_string: str) -> bool:
    original_structure = get_database_structure(connection_string)
    
    inverted_data = inversion_db(mapping_rules, actual_output, connection_string)
    
    inverted_structure = {}
    for table_name, csv_data in inverted_data.items():
        sql = create_sql_from_csv(csv_data, table_name)
        # Here you would typically execute this SQL to create the inverted database
        # For now, we'll just parse it to get the structure
        df = pd.read_csv(pd.compat.StringIO(csv_data))
        inverted_structure[table_name] = {col: 'TEXT' for col in df.columns}
    
    return compare_database_structures(original_structure, inverted_structure)