# Load tests

# Iterate tests
## Find RML rules
For each triple:
* subject
  * map
    * type
      * template (Eg. http://example.com/{ID}/{Name})
    * value
  * term type
* predicate
  * map
    * type (= constant, always?)
    * value
* object
  * map
    * type
      * reference (eg. Name)
      * template (eg. {Name} {LastName})
      * constant (eg. http://example.com/Sport)
    * value
      * in format of the (eg.) in map.type
  * termtype
    * Literal
    * IRI
## Generate sparql endpoint
### Create
### Load data
## Find subjects in endpoint
