# Inverting knowledge graph mappings

A master thesis by Tijs Van Kampen

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

Build the pyrdf4j package and install it. Additionally, a triplestore implementing the [rdf4j API](https://rdf4j.org/documentation/reference/rest-api/) is required. This comes down to implementing both a [SPARQL endpoint](https://www.w3.org/TR/sparql11-protocol/) and the [SPARQL Graph Store HTTP Protocol](https://www.w3.org/TR/sparql11-http-rdf-update/). For this project, [GraphDB](https://graphdb.ontotext.com/) was used, but any triplestore implementing these protocols, ~~like blazegraph~~[deprecated], should work. For now, the triplestore is assumed to be running on localhost:7200, in later versions this will be configurable.


# TODO's:
## Implementation
- [ ] Generate the source files (basic version)
- [ ] Add the option to use RDFLib's build-in triplestore instead of pyrdf4j (it offers both in-memory and on-disk stores)*
  - [ ] Need to check how supporting blank nodes would work here
- [ ] Mapping the graph 
  - [ ] For smarter data extraction
  - [ ] For join conditions

## Paper
- [ ] Create general structure
- [ ] Write introduction
- [ ] Write related work
- [ ] Write limitations (entropy in join conditions, blank nodes, ...) [maybe this should be in the implementation section]
- [ ] Write implementation
  - [ ] High-level algorithm
- [ ] Evaluation
  - [ ] Test cases
  - [ ] Larger experiments
  - [ ] Results
  - [ ] Discussion
- [ ] Conclusion
  - [ ] Future work