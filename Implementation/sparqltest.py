from SPARQLWrapper import SPARQLWrapper, JSON
import json
import sys, os

sys.stdout = open("outsparql2.txt", "w")

sparql = SPARQLWrapper("http://DESKTOP-IV4QGIH:7200/repositories/PWC-morphed")
sparql.setReturnFormat(JSON)

sparql.setQuery("""
SELECT DISTINCT ?s
WHERE {
    ?s ?p ?o
}
""")

qres = sparql.query().convert()["results"]["bindings"]

subjects = [x["s"]["value"] for x in qres]

print(len(subjects))
print(json.dumps(subjects, indent=4))