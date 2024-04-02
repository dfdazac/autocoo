#load libraries
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from Bio_Epidemiology_NER.bio_recognizer import ner_prediction

from sentence_transformers import SentenceTransformer
# from word2number import w2n
# import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from owlready2 import *
import argparse
import types
import re
import os


labels_cancer = ["Measure of quality of life",
    "Measure of patient satisfaction",
    "Measure of tolerability",
    "Measure of adherence",
    "Measure of costs",
    "Measure of efficacy",
    "Measure of healthcare resource utilization",
    "Measure of neurocognitive function",
    "Measure of nutritional status",
    "Measure of patient preference",
    "Measure of performance status",
    "Measure of sexual function",
    "Measure of safety"]

labels_default = []

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--csv', 
                        help="absolute path to csv file with data. Needs to include the following columns: ['Primary Outcome Measures','Secondary Outcome Measures','Other Outcome Meausures','nct_id'].",
                        default="git/autocoo/data/debugging-studies.csv")
    
    parser.add_argument('--resultfolder', 
                        help="absolute path to folder where NER and ontologies will be saved.",
                        default="git/autocoo/data/debuggin")

    parser.add_argument('--process', 
                        help="one of [semi,auto]. Semi will return a list of entities detected in the csv file of clinical trials. Auto will return an ontology engineered from the csv file containing clinical trials.",
                        default='auto')
    
    parser.add_argument('--cutoff', 
                        help="hierarchical clustering cut-off value. 0 (default) means no hierarchichal clustering will be applied, 0.3 is the cutuff we used in our work in the automatic procedure. Ignored for semi.",
                        default=0.3)

    args = parser.parse_args()

    # parse csv data
    data = pd.read_csv(args.csv)
    
    # run NER
    ner_result = ner_on_df(data)

    # save NER results
    os.makedirs(args.resultfolder, exist_ok=True)
    resfile = os.path.join(args.resultfolder,"NER_results.csv")
    ner_result.to_csv(resfile)

    # extract only the values that we want to put into the ontology in the automatic procedures
    outcome_measures = filter_ner_result(ner_result)


    # run clustering
    clustered = hierarchical_clustering(outcome_measures,cutoff=args.cutoff)

    # run LDA
    result = topic_categosization(outcome_measures,clustered)
    
    # run engineering and save ontology
    onto = ontology(result,args.resultfolder,args.cutoff)


# iterates through a dataframe of outcome measure
# takes hard-coded column names to retrieve the outcome measures and study id from dataframe 
def ner_on_df(data):
    result = pd.DataFrame(columns = ['entity_group', 'value', 'score', 'from','nct_id'])

    for i,row in data.iterrows():
        # print(row)
        out = pd.DataFrame(columns = ['entity_group', 'value', 'score', 'from'])

        if isinstance(row['Primary Outcome Measures'],str):
            for split in row['Primary Outcome Measures'].split('|'):
                out1 = ner_prediction(corpus=split,compute='cpu') #pass compute='gpu' if using gpu
                out1['from'] = 'primary'
                # print(out1)
                
                out = out.append(out1,ignore_index=True)

            # print(out)
            # print(row['Primary Outcome Measures'])
        
        if isinstance(row['Secondary Outcome Measures'],str):
            for split in row['Secondary Outcome Measures'].split('|'):
                out1 = ner_prediction(corpus=split,compute='cpu') #pass compute='gpu' if using gpu
                out1['from'] = 'secondary'
                # print(out1)
                out = out.append(out1,ignore_index=True)
            # print(row['Secondary Outcome Measures'])
        
        
        if isinstance(row['Other Outcome Measures'],str):
            for split in row['Secondary Outcome Measures'].split('|'):
                out1 = ner_prediction(corpus=split,compute='cpu') #pass compute='gpu' if using gpu
                out1['from'] = 'other'
                # print(out1)
                out = out.append(out1,ignore_index=True)

        out['nct_id'] = row['nct_id']
        result = result.append(out,ignore_index=True)
        # print(result)
        # break
    
    return result


def hierarchical_clustering(outcome_measures,cutoff=0):
    print(outcome_measures)
    
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(outcome_measures)
    Z = linkage(sentence_embeddings, metric='cosine', method='average')

    clusters = fcluster(Z, t=cutoff, criterion='distance')

    outcome_measures_clustered = pd.DataFrame({'Outcome Measure': outcome_measures, 'Cluster': clusters})
    return outcome_measures_clustered


def topic_categosization(data,clusters):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    # return "not implemented"
    # Prepare data for LDA
    vectorizer_outcome = CountVectorizer()
    vectorizer_labels = CountVectorizer()

    X_outcome = vectorizer_outcome.fit_transform(data)
    labels = labels_cancer
    X_labels = vectorizer_labels.fit_transform(labels)

    # Fit LDA models
    lda_model_outcome = LatentDirichletAllocation(n_components=len(labels), random_state=0)
    lda_model_outcome.fit(X_outcome)
    topics = lda_model_outcome.transform(X_outcome)

    lda_model_labels = LatentDirichletAllocation(n_components=len(labels), random_state=0)  # assuming each label represents a different topic
    lda_model_labels.fit(X_labels)

    # Extract keywords from outcome measures
    feature_names_outcome = vectorizer_outcome.get_feature_names_out()
    keywords_per_topic_outcome = []
    for topic_idx, topic in enumerate(lda_model_outcome.components_):
        keywords_per_topic_outcome.append([feature_names_outcome[i] for i in topic.argsort()[:-6:-1]])

    # Extract keywords from labels
    feature_names_labels = vectorizer_labels.get_feature_names_out()
    keywords_per_topic_labels = []
    for topic_idx, topic in enumerate(lda_model_labels.components_):
        keywords_per_topic_labels.append([feature_names_labels[i] for i in topic.argsort()[:-6:-1]])

    # Calculate similarity between the extracted keywords from outcomes and labels
    similarity_scores_per_cluster = []
    for cluster_id in clusters['Cluster'].unique():
        cluster_topic_idx = topics[clusters[clusters['Cluster'] == cluster_id].index[0]].argmax()
        cluster_keywords = keywords_per_topic_outcome[cluster_topic_idx]
        cluster_keywords_embedding = model.encode(cluster_keywords)
        cluster_keywords_embedding_mean = np.mean(cluster_keywords_embedding, axis=0)
        
        label_similarity_scores = []
        for label_keywords in keywords_per_topic_labels:
            label_keywords_embedding = model.encode(label_keywords)
            label_keywords_embedding_mean = np.mean(label_keywords_embedding, axis=0)
            label_similarity_scores.append(cosine_similarity([cluster_keywords_embedding_mean], [label_keywords_embedding_mean])[0][0])

        similarity_scores_per_cluster.append(label_similarity_scores)

    # Assign labels based on the highest similarity score and a threshold of 0.5
    label_threshold = 0.5

    labels_per_outcome = []
    for cluster_id in clusters['Cluster']:
        similarity_scores = np.array(similarity_scores_per_cluster[cluster_id-1])
        max_score_idx = similarity_scores.argmax()
        max_score = similarity_scores[max_score_idx]
        if max_score > label_threshold:
            labels_per_outcome.append(labels[max_score_idx])

    # Add the labels column to the outcome_measures_clustered dataframe
    clusters['Label'] = labels_per_outcome

    # Merge the clusters using the pandas concat function
    merged_clusters = pd.concat([clusters.groupby('Label')['Outcome Measure'].agg(list),  clusters.groupby('Label').size()], axis=1).reset_index()
    merged_clusters.columns = ['Label', 'Outcome Measure', 'Size']

    # Print the merged clusters
    return merged_clusters


def ontology(data,folder,cutoff):
    # Create a new ontology
    ontology = get_ontology("http://example.com/ontology.owl")

    desired_labels = labels_cancer

    with ontology:
        # For each desired label
        for label in desired_labels:
            # Clean up the label name to be a valid class name
            label_class_name = re.sub('\W|^(?=\d)','_', label)
            # Create a new OWL class for this label
            LabelClass = types.new_class(label_class_name, (Thing,))
            # Append original label name as rdfs:label
            LabelClass.label.append(label)
            # Get the outcome measures for this label
            cluster_outcomes = data[data['Label'] == label]['Outcome Measure']
            # For each outcome measure list
            for outcome_list in cluster_outcomes:
                # For each outcome in the list
                for outcome in outcome_list:
                    # Clean up the outcome name to be a valid class name
                    outcome_class_name = re.sub('\W|^(?=\d)','_', outcome)
                    # Create a new OWL class for this outcome measure, as a subclass of LabelClass
                    OutcomeClass = types.new_class(outcome_class_name, (LabelClass,))
                    # Append original outcome name as rdfs:label
                    OutcomeClass.label.append(outcome)

    # Save the ontology to an OWL file
    ontology.save(file = os.path.join(folder,"automatic_"+str(cutoff)+"_ontology.owl"), format = "rdfxml")


def filter_ner_result(ner_result):
    filtered_df = ner_result.loc[ner_result['entity_group'] == 'Diagnostic_procedure']
    measures = list(set(filtered_df['value']))
    
    outcome_measures = []
    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]') 

    for m in measures:
        split_m = []
        for xs in m.split(" "):
            for x in xs.split("."):
                split_m.append(x)

        # remove duplicate words while preserving order
        split_m = list(dict.fromkeys(split_m))
        new_m =""

        for s in split_m:
            if(regex.search(s) == None ):
                new_m = new_m + " " +s
        
        if not new_m == "":
            if outcome_measures == None:
                outcome_measures = [new_m.strip()]
            else:
                outcome_measures.append(new_m.strip())
        
    # print(outcome_measures)
    return outcome_measures


if __name__ == '__main__':
    main()
    




