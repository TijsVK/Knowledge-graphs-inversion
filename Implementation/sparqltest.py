from SPARQLWrapper import SPARQLWrapper, JSON
import json
import sys, os

sys.stdout = open("outsparql2.txt", "w")

sparql = SPARQLWrapper("http://publications.europa.eu/webapi/rdf/sparql")
sparql.setReturnFormat(JSON)

get_all_subjects_query = """
SELECT DISTINCT ?s
WHERE {
    ?s ?p ?o
}
"""

sparql.setQuery(get_all_subjects_query)

qres = sparql.query().convert()["results"]["bindings"]

subjects = [x["s"]["value"] for x in qres]

with open("subjects.json", "w") as f:
    json.dump(subjects, f, indent=4)