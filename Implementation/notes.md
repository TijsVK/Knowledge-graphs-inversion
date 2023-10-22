# Extracting data to RDF with RML
Test cases

## XPath

## JSONPath

## SQL

### Source
#### full table
eg. 
```
rml:logicalSource [
    rml:source <#DB_source>;
    rr:sqlVersion rr:SQL2008;
    rr:tableName "student";
];
```

#### query
eg.
```
rml:logicalSource [
    rml:source <#DB_source>;
    rr:sqlVersion rr:SQL2008;
    rml:query """
        SELECT ID, FirstName, LastName
        FROM Employee
        WHERE ID < 30
    """ ;
    rml:referenceFormulation ql:CSV
];
```
Conditions => ignore?

### Referencing

Works as a CSV file


# Extracting data from RDF with RML

## TriplesMap

### Source

#### CSV
`rml:referenceFormulation ql:CSV`
is used for both actual CSV (/TSV/...) files and for tables in a database.

Using this formulation, the source iterates over the rows of the CSV file. As such we don't need to mind any nesting or wildcards.

### Subject predicate object

#### Subject
Can be done with a subject map `rr:subjectMap` or a constant shortcut `rr:subject`.

#### Predicate
Can be done with a predicate map `rr:predicateMap` or a constant shortcut `rr:predicate`.

#### Object
Can be done with an object map `rr:objectMap` or a constant shortcut `rr:object`.

### Generating RDF terms

#### Map vs constant
Map has to be used when the value is variable. Constant can be used when the value is constant. For any iteration, the value for the constant is the same. 

eg.
```
<http://trans.example.com/stop/645> rdf:type ex:Stop.
<http://trans.example.com/stop/651> rdf:type ex:Stop.
<http://trans.example.com/stop/873> rdf:type ex:Stop.
```
When generating this data the subject is variable for each value of the iteration. The predicate and object however are constant and thus can be mapped with a constant shortcut.

rml:
```
rr:subjectMap [
    rr:template
      "http://trans.example.com/stop/{@id}";
];
rr:predicate ex:type;
rr:object ex:Stop;
```