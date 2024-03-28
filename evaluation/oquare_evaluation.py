import rdflib
from rdflib import Graph, Literal, Namespace,  BNode, RDF, URIRef
from rdflib.plugins.sparql import prepareQuery
import sys
import argparse

# Define the OWL, RDF, and SKOS namespaces
owl = Namespace("http://www.w3.org/2002/07/owl#")
rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
skos = Namespace("http://www.w3.org/2004/02/skos/core#")


def get_classes_count(graph):
    #Query to obtain a class
    query = """
        SELECT (COUNT(?class) AS ?count)
        WHERE {
            ?class a owl:Class .
        }
    """

    # Execute the SPARQL query
    results = graph.query(query)

    # Process the query results
    for row in results:
        class_count = row["count"]
    
    print("Number of classes:", class_count)
    classes_count = class_count
    return class_count


def nomonto(graph):
    #Create a dictionary to store the property count for each class
    class_properties = {}

    # Query for each class and count the properties
    query = """
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT ?class (COUNT(?property) AS ?propertiesCount)
        WHERE {
            ?class a owl:Class .
            ?class ?property ?value .
            FILTER (isURI(?property))
        }
        GROUP BY ?class
    """
    results = graph.query(query, initNs={"owl": owl})
    for row in results:
        class_uri = row["class"].toPython()
        properties_count = int(row["propertiesCount"].value)  # Convert Literal to integer
        class_properties[class_uri] = properties_count

    # Calculate the sum of properties count
    properties_sum = sum(class_properties.values())
    # print(properties_sum)

    # Calculate the sum of class count
    classes_count = len(list(graph.subjects(predicate=RDF.type, object=owl.Class)))

    # Calculate the NOMOnto metric
    NOMOnto = properties_sum / classes_count

    # Print the calculated NOMOnto value
    print("NOMOnto (Number of Properties):", NOMOnto)
    return NOMOnto


def wmconto(graph):
    query_properties_relationships_per_class = prepareQuery(
        """
        SELECT ?class (COUNT(?property) AS ?propertiesCount) (COUNT(?relationship) AS ?relationshipsCount)
        WHERE {
            ?class rdf:type owl:Class .
            OPTIONAL { ?class ?property ?value FILTER(isIRI(?value)) } .
            OPTIONAL { ?subject ?relationship ?class FILTER(isIRI(?subject)) } .
        }
        GROUP BY ?class
        """,
        initNs={"rdf": Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#"), "owl": Namespace("http://www.w3.org/2002/07/owl#")}
    )

    # Execute the SPARQL query
    properties_relationships_per_class = graph.query(query_properties_relationships_per_class)

    # Count the total number of properties, relationships, and classes
    total_properties = 0
    total_relationships = 0
    num_classes = 0

    for result in properties_relationships_per_class:
        total_properties += int(result['propertiesCount'].value)
        total_relationships += int(result['relationshipsCount'].value)
        num_classes += 1

    # Calculate WMCOnto
    if num_classes > 0:
        WMCOnto = (total_properties + total_relationships) / num_classes
    else:
        WMCOnto = 0

    print("Weighted Method Count (WMCOnto):", WMCOnto)
    return WMCOnto

def inronto(graph):
    # SPARQL query to count the relationships for each class
    query_relationships_per_class = prepareQuery(
        """
        SELECT ?class (COUNT(?relationship) AS ?relationshipCount)
        WHERE {
            ?class rdf:type owl:Class .
            ?class ?relationship ?object .
            FILTER (?relationship != rdf:type && ?relationship != rdfs:subClassOf) .
        }
        GROUP BY ?class
        """
        ,
        initNs={"rdf": Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#"), "owl": Namespace("http://www.w3.org/2002/07/owl#"), "rdfs": Namespace("http://www.w3.org/2000/01/rdf-schema#")}
    )

    # Execute the SPARQL query to count the relationships per class
    relationships_per_class = graph.query(query_relationships_per_class)

    # Calculate the total number of relationships and the total number of classes
    total_relationships = 0
    total_classes = 0

    for result in relationships_per_class:
        relationship_count = int(result['relationshipCount'].value)
        total_relationships += relationship_count
        total_classes += 1

    # Calculate INROnto
    if total_classes > 0:
        INROnto = total_relationships / total_classes
    else:
        INROnto = 0

    print("Relationships per class (INROnto):", INROnto)
    return INROnto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ontology', 
                        help="absolute path to ontology for which the OQuaRE metrics should be calculated.",
                        required=True)

    args = parser.parse_args()

    graph = Graph()
    graph.parse(args.ontology)

    ccount = get_classes_count(graph)
    NOMOnto = nomonto(graph)
    WMCOnto = wmconto(graph)
    INROnto = inronto(graph)

if __name__ == '__main__':
    main()
    




