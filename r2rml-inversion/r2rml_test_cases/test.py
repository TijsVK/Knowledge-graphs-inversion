import os
import sys
import csv
import mysql.connector
import psycopg2
from configparser import ConfigParser, ExtendedInterpolation
from rdflib import ConjunctiveGraph, RDF, Namespace, compare, Literal, URIRef

mysql_exceptions = ["R2RMLTC0002d", "R2RMLTC0003b", "R2RMLTC0014a", "R2RMLTC0014b", "R2RMLTC0014c"]
mysql_non_compliance = ["R2RMLTC0002f", "R2RMLTC0018a"]


def test_all():
    q1 = """SELECT ?database_uri WHERE { 
        ?database_uri rdf:type <http://purl.org/NET/rdb2rdf-test#DataBase>. 
      } ORDER BY ?database_uri"""
    for r in manifest_graph.query(q1):
        database_uri = r.database_uri
        d_identifier = manifest_graph.value(subject=database_uri, predicate=DCELEMENTS.identifier, object=None)
        d_identifier = d_identifier.toPython()
        d_title = manifest_graph.value(subject=database_uri, predicate=DCELEMENTS.title, object=None)
        d_title = d_title.toPython()
        database = manifest_graph.value(subject=database_uri, predicate=RDB2RDFTEST.sqlScriptFile, object=None)
        database = database.toPython()
        print("**************************************************************************")
        print("Using the database: " + d_identifier + " (" + d_title + ")")
        database_load(database)
        q2 = """SELECT ?test_uri WHERE { 
                ?test_uri <http://purl.org/NET/rdb2rdf-test#database> ?database_uri. 
              } ORDER BY ?test_uri"""
        for r2 in manifest_graph.query(q2, initBindings={'?database_uri': URIRef(database_uri)}):
            test_uri = r2.test_uri
            t_identifier = manifest_graph.value(subject=test_uri, predicate=DCELEMENTS.identifier, object=None)
            t_identifier = t_identifier.toPython()
            t_title = manifest_graph.value(subject=test_uri, predicate=DCELEMENTS.title, object=None)
            t_title = t_title.toPython()
            purpose = manifest_graph.value(subject=test_uri, predicate=TESTDEC.purpose, object=None)
            purpose = purpose.toPython()
            expected_output = manifest_graph.value(subject=test_uri, predicate=RDB2RDFTEST.hasExpectedOutput, object=None)
            expected_output = expected_output.toPython()
            r2rml = manifest_graph.value(subject=test_uri, predicate=RDB2RDFTEST.mappingDocument, object=None)
            r2rml = r2rml.toPython()
            print("-----------------------------------------------------------------")
            print("Testing R2RML test-case: " + t_identifier + " (" + t_title + ")")
            print("Purpose of this test is: " + purpose)
            run_test(t_identifier, r2rml, test_uri, expected_output)


def test_one(identifier):
    test_uri = manifest_graph.value(subject=None, predicate=DCELEMENTS.identifier, object=Literal(identifier))
    t_title = manifest_graph.value(subject=test_uri, predicate=DCELEMENTS.title, object=None)
    t_title = t_title.toPython()
    purpose = manifest_graph.value(subject=test_uri, predicate=TESTDEC.purpose, object=None)
    purpose = purpose.toPython()
    expected_output = manifest_graph.value(subject=test_uri, predicate=RDB2RDFTEST.hasExpectedOutput, object=None)
    expected_output = expected_output.toPython()
    r2rml = manifest_graph.value(subject=test_uri, predicate=RDB2RDFTEST.mappingDocument, object=None)
    r2rml = r2rml.toPython()
    database_uri = manifest_graph.value(subject=test_uri, predicate=RDB2RDFTEST.database, object=None)
    database = manifest_graph.value(subject=database_uri, predicate=RDB2RDFTEST.sqlScriptFile, object=None)
    database = database.toPython()
    print("Testing R2RML test-case: " + identifier + " (" + t_title + ")")
    print("Purpose of this test is: " + purpose)
    database_load(database)
    run_test(identifier, r2rml, test_uri, expected_output)


def database_up():
    if database_system == "mysql":
        os.system("docker compose -f databases/docker-compose-mysql.yml stop")
        os.system("docker compose -f databases/docker-compose-mysql.yml rm --force")
        os.system("docker compose -f databases/docker-compose-mysql.yml up -d && sleep 30")
    elif database_system == "postgresql":
        os.system("docker compose -f databases/docker-compose-postgresql.yml stop")
        os.system("docker compose -f databases/docker-compose-postgresql.yml rm --force")
        os.system("docker compose -f databases/docker-compose-postgresql.yml up -d && sleep 30")


def database_down():
    if database_system == "mysql":
        os.system("docker compose -f databases/docker-compose-mysql.yml stop")
        os.system("docker compose -f databases/docker-compose-mysql.yml rm --force")
    elif database_system == "postgresql":
        os.system("docker compose -f databases/docker-compose-postgresql.yml stop")
        os.system("docker compose -f databases/docker-compose-postgresql.yml rm --force")


def database_load(database_script):
    print("Loading in " + config["properties"]["database_system"] + " system the file:" + database_script)

    if database_system == "mysql":
        host = os.environ.get('HOST', '127.0.0.1')
        cnx = mysql.connector.connect(user='r2rml', password='r2rml', host=host, database='r2rml')
        cursor = cnx.cursor()
        for statement in open('databases/' + database_script):
            cursor.execute(statement)
        cnx.commit()
        cursor.close()
        cnx.close()

    elif database_system == "postgresql":
        host = os.environ.get('HOST', 'localhost')
        cnx = psycopg2.connect("dbname='r2rml' user='r2rml' host='" + host + "' password='r2rml'")
        cursor = cnx.cursor()
        if database_script == "d016.sql":
            database_script = "d016-postgresql.sql"
        for statement in open('databases/' + database_script):
            cursor.execute(statement)
        cnx.commit()
        cursor.close()
        cnx.close()


def run_test(t_identifier, mapping, test_uri, expected_output):
    if database_system == "mysql" and t_identifier in mysql_exceptions:
        mapping = mapping.replace(".ttl", "-mysql.ttl")

    if database_system == "mysql" and t_identifier in mysql_non_compliance:
        print(f"Skipped test {t_identifier} because MySQL non-compliance with ANSI SQL")
        results.append([config["tester"]["tester_name"], config["engine"]["engine_name"], get_database_name(), t_identifier, "untested"])
        return

    os.system("cp " + t_identifier + "/" + mapping + " r2rml.ttl")
    expected_output_graph = ConjunctiveGraph()
    if os.path.isfile(config["properties"]["output_results"]):
        os.system("rm " + config["properties"]["output_results"])

    if expected_output:
        output = manifest_graph.value(subject=test_uri, predicate=RDB2RDFTEST.output, object=None)
        output = output.toPython()
        expected_output_graph.parse("./" + t_identifier + "/" + output, format="nquads")

    exit_code = os.system(
        config["properties"]["engine_command"] + " > " + t_identifier + "/engine_output-" + database_system + ".log")

    # if there is output file
    if os.path.isfile(config["properties"]["output_results"]):
        extension = config["properties"]["output_results"].split(".")[-1]
        os.system("cp " + config["properties"]["output_results"] + " " +
                t_identifier + "/engine_output-" + database_system + "." + extension)
        # and expected output is true
        if expected_output:
            output_graph = ConjunctiveGraph()
            iso_expected = compare.to_isomorphic(expected_output_graph)
            # trying to parse the output (e.g., not valid RDF)
            try:
                output_graph.parse(config["properties"]["output_results"],
                                   format=config["properties"]["output_format"])
                iso_output = compare.to_isomorphic(output_graph)
                # and graphs are equal
                if iso_expected == iso_output:
                    result = passed
                # and graphs are distinct
                else:
                    print("Output RDF does not match with the expected RDF")
                    result = failed
            # output is not valid RDF
            except:
                print("Output RDF is invalid")
                result = failed

        # and expected output is false and error-code
        elif exit_code != 0:
            print("The processor returned a non-zero error code signalling a mistake (even though a file might have been generated)")
            result = passed
        # and expected output is false
        else:
            print("Output RDF found but none was expected")
            result = failed
    # if there is no output file
    else:
        # and expected output is true
        if expected_output:
            print("No RDF output found while output was expected")
            result = failed
        # expected output is false
        else:
            result = passed

    results.append(
        [config["tester"]["tester_name"], config["engine"]["engine_name"], get_database_name(), t_identifier, result])
    print(t_identifier + "," + result)


def generate_results():
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)

    print("Generating the RDF results using EARL vocabulary")
    os.system("java -jar rmlmapper.jar -m mapping.rml.ttl -o results-" + database_system + ".ttl -d")
    os.system("rm metadata.csv r2rml.ttl && mv results.csv results-" + database_system + ".csv")


def merge_results():
    if os.path.isfile("results-mysql.ttl") and os.path.isfile("results-postgresql.ttl"):
        final_results = ConjunctiveGraph()
        final_results.parse("results-mysql.ttl", format="ntriples")
        final_results.parse("results-postgresql.ttl", format="ntriples")
        final_results.serialize("results.ttl", format="ntriples")
    elif os.path.isfile("results-mysql.ttl"):
        final_results = ConjunctiveGraph()
        final_results.parse("results-mysql.ttl", format="ntriples")
        final_results.serialize("results.ttl", format="ntriples")
    elif os.path.isfile("results-postgresql.ttl"):
        final_results = ConjunctiveGraph()
        final_results.parse("results-postgresql.ttl", format="ntriples")
        final_results.serialize("results.ttl", format="ntriples")


def get_database_url():
    if database_system == "mysql":
        return "https://www.mysql.com/"
    elif database_system == "postgresql":
        return "https://www.postgresql.org/"
    else:
        print("Database system declared in config file must be mysql or postgresql")
        sys.exit()


def get_database_name():
    if database_system == "mysql":
        return "MySQL"
    elif database_system == "postgresql":
        return "PostgreSQL"
    else:
        print("Database system declared in config file must be mysql or postgresql")
        sys.exit()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Configuration file is missing: python3 test.py <config file>")
        sys.exit(1)

    config_file = str(sys.argv[1])
    if not os.path.isfile(config_file):
        print("The configuration file " + config_file + " does not exist.")
        print("Aborting...")
        sys.exit(2)

    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(config_file)
    database_system = config["properties"]["database_system"]

    manifest_graph = ConjunctiveGraph()
    manifest_graph.parse("./manifest.ttl", format='turtle')
    RDB2RDFTEST = Namespace("http://purl.org/NET/rdb2rdf-test#")
    TESTDEC = Namespace("http://www.w3.org/2006/03/test-description#")
    DCELEMENTS = Namespace("http://purl.org/dc/terms/")

    results = [["tester", "platform", "rdbms", "testid", "result"]]
    metadata = [
        ["tester_name", "tester_url", "tester_contact", "test_date", "engine_version", "engine_name", "engine_created",
         "engine_url", "database", "database_name"],
        [config["tester"]["tester_name"], config["tester"]["tester_url"], config["tester"]["tester_contact"],
         config["engine"]["test_date"],
         config["engine"]["engine_version"], config["engine"]["engine_name"], config["engine"]["engine_created"],
         config["engine"]["engine_url"], get_database_url(), get_database_name()]]
    failed = "failed"
    passed = "passed"
    with open('metadata.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(metadata)

    print("Deployment docker container for " + database_system + "...")
    database_up()

    if config["properties"]["tests"] == "all":
        test_all()
        generate_results()
    else:
        test_one(config["properties"]["tests"])
        generate_results()

    database_down()
    merge_results()
