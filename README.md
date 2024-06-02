# Inverting knowledge graph mappings

A monorepo for a master thesis by Tijs Van Kampen. Contains both the implementation and the paper

## Getting started

Clone the repository with the --recurse-submodules flag to clone the required submodules as well.

```bash
git clone --recurse-submodules https://github.com/TijsVK/Knowledge-graphs-inversion
```

It is recommended to use a virtual environment to install the required packages. Preferences as to what to use vary, so no exact instructions or environments besides a requirements.txt file in /Implementation are provided.

Install the base requirements.

```bash
pip install -r ./Implementation/requirements.txt
```

If you want to invert local graph files, run initialization.py (which adds some config files to pyrdf4j).

```bash
python ./Implementation/initialise.py
```

Build the pyrdf4j package and install it. To invert local rdf files, a triplestore implementing the [rdf4j API](https://rdf4j.org/documentation/reference/rest-api/) is required. This comes down to implementing both a [SPARQL endpoint](https://www.w3.org/TR/sparql11-protocol/) and the [SPARQL Graph Store HTTP Protocol](https://www.w3.org/TR/sparql11-http-rdf-update/). For this project, [GraphDB](https://graphdb.ontotext.com/) was used, but any triplestore implementing these protocols, ~~like blazegraph~~[deprecated], should work. The triplestore is assumed to be running on localhost:7200.


# Known issues
- Same-type referencing joins do not work


# Roadmap
- Update demo with latest features
  - JSON sources
  - Remote sources