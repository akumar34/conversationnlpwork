import argparse, sys, ontology_reader, dataset_walker, time, json, nltk
from collections import OrderedDict
from collections import Counter
import re, string
from nltk import word_tokenize as wt
import kenlm
import numpy as np

dataset = dataset_walker.dataset_walker("dstc4_train", dataroot="data", labels=True)

def generate_feature_vector(bigrams, bigrams_data):
    feature_vector = [bigrams_data[bigram] for bigram in bigrams]
    return feature_vector

#tagsets = ontology_reader.OntologyReader(args.ontology).get_tagsets()

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def preprocessor(utterance):
    pause_acts = re.findall('%[\w]*',utterance)
    for act in pause_acts:
        utterance = utterance.replace(act,'').strip()

    pause_acts = re.findall('[\w]*-',utterance)
    for act in pause_acts:
        utterance = utterance.replace(act,'').strip()

    pause_acts = re.findall('[\w]*~',utterance)
    for act in pause_acts:
        utterance = utterance.replace(act,'').strip()

    parsed_utterance = ''
    for word in utterance.split():
        parsed_utterance += word + ' '

    parsed_utterance = parsed_utterance.strip()

    return parsed_utterance.lower()


def tokenizer(utterance):
    return wt(utterance.strip())

V = CountVectorizer(analyzer='word',preprocessor=preprocessor, tokenizer=tokenizer, ngram_range=(2,2), lowercase=True, binary=False)
vocab=Counter()
corpus=[]
topics=set()
for call in dataset:
    for utter, label in call:
        topic = utter['segment_info']['topic']
        topics.add(topic)
        speaker = utter['speaker']
        utterance = utter['transcript']
        vocab.update(preprocessor(utterance))
        corpus.append((speaker, topic, utterance))

X = V.fit_transform([utterance for _,_,utterance in corpus])

n_clusters=len(topics)-1

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

cluster_map = dict()
for i in range(0,len(kmeans.labels_)):
    label = kmeans.labels_[i]
    utterance = corpus[i][2]
    if label not in cluster_map:
        cluster_map[label] = []
    cluster_map[label].append(utterance)

emission_probs = dict()
smooth_delta = 1.0
for key in cluster_map.keys():
    unigrams_freq = Counter()
    bigrams_freq = Counter()
    for utterance in cluster_map[key]:
        processed_utterance = tokenizer(utterance)
        unigrams_freq.update(processed_utterance)
        bigrams_freq.update(nltk.bigrams(processed_utterance))
    if key not in emission_probs:
        emission_probs[key] = dict()
    for utterance in cluster_map[key]:        
        processed_utterance = tokenizer(utterance)
        bigrams = nltk.bigrams(processed_utterance)
        for bigram in bigrams:
            word1 = bigram[0]
            word2 = bigram[1]
            if bigram not in emission_probs[key]:
                emission_probs[key][bigram] = 0.0
            emission_probs[key][bigram] = (bigrams_freq[bigram] + smooth_delta) / (unigrams_freq[word1] + smooth_delta*len(vocab))
            

