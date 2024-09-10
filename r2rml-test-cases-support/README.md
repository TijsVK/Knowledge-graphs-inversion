# R2RML Test-Cases support

Test the capabilities of your R2RML engine with the [R2RML test cases](https://www.w3.org/2001/sw/rdb2rdf/test-cases/). Use the resources provided in this repository to automatically generate an EARL report with your results. Go to the [R2RML implementation report](https://github.com/kg-construct/r2rml-implementation-report/) repository to see how to include the generated report.

**IMPORTANT INFORMATION ABOUT THE R2RML IMPLEMENTATION REPORT:** 
- This repository does NOT include any engine report
- The support resources provided here can be: forked, downloaded or cloned but no PR with the report is needed. A recommendation to maintain up-to-date the resources of this repository is to add a submodule directly to your github repo. 
- Read carefully the documentation provided, and open an issue if you have any question or doubt.

## Requirements for creating the EARL implementation report:

- Linux based OS
- Docker and docker-compose
- Python
- Java

## RDBMS coverage and properties info:

- MySQL v8.0 (`port = 3306`)
- PostgreSQL v13.4 (`port = 5432`)

Connection properties for any RDBMS are: `database = r2rml, user = r2rml, password = r2rml`.

For testing purposes, **mapping path is invariable, it is always `./r2rml.ttl`** and **the base IRI of the output is http://example.com/base/**


## Steps to generate the results from the R2RML test-cases:

1. Create a submodule (recommended) or fork/clone/download this repository.
2. Include a way to run your engine with the resources of this folder.
3. Install the requirements of the script `python3 -m pip install -r requirements.txt`
4. Modify the config.ini file with your information. For the correspoding configurating of your R2RML engine, remember that the path of the **mapping file is always ./r2rml.ttl**. For example:

```
[tester]
tester_name: David Chaves # tester name
tester_url: https://dchaves.oeg-upm.net/ # tester homepage
tester_contact: dchaves@fi.upm.es # tester contact

[engine]
test_date: 2021-01-07 # engine test-date (YYYY-MM-DD)
engine_version: 3.12.5 # engine version
engine_name: Morph-RDB # engine name
engine_created: 2013-12-01 # engine date created (YYYY-MM-DD)
engine_url: https://morph.oeg.fi.upm.es/tool/morph-rdb # URL of the engine (e.g., GitHub repo)


[properties]
database_system: [mysql|postgresql] # choose only one
output_results: ./output.ttl # path to the result graph of your engine
output_format: ntriples # output format of the results from your engine
engine_command: java -jar morph-rdb.jar -p properties.properties # command to run your engine
```

5. Run the script `python3 test.py config.ini`
6. Your results will appear in `results.ttl` in RDF and in `results.csv` in CSV.
7. Upload or update the obtained results the access point you have provided in the configuration step.
8. For each new version of your engine, repeat the process from step 4 to 7.


Overview of the testing steps:
![Testing setp](misc/test.png?raw=true "Testing setp")


### Notes
- The MySQL Docker container stores timestamps as UTC. Values that are retrieved from the database may therefore not correspond with the time zone of the host. If you notice that a test fails because the times are off, try appending `?useLegacyDatetimeCode=false&serverTimezone=XXX`, where `XXX` corresponds with your [IANA Time Zone Database](https://www.iana.org/time-zones) entry (e.g., `Europe/Brussels`), to the JDBC connection URL.
