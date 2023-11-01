from SPARQLWrapper import SPARQLWrapper, JSON
import json
import sys, os

sys.stdout = open("outsparql2.txt", "w")

sparql = SPARQLWrapper("http://DESKTOP-IV4QGIH:7200/repositories/PWC-morphed")
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

print(len(subjects))
print(json.dumps(subjects, indent=4))