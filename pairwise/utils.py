from ast import Pass
import math
from itertools import combinations
from math import isclose
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import wordnet
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import wordnet, pos_tag
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer

def calculate_wup_similarity(tokens1, tokens2):
    # Calculate the maximum Wu-Palmer similarity between each pair of synsets
    max_similarity = 0.0
    for token1 in tokens1:
        for token2 in tokens2:
            synset1 = wordnet.synsets(token1)
            synset2 = wordnet.synsets(token2)
            if synset1 and synset2:
                similarity = wordnet.wup_similarity(synset1[0], synset2[0])
                if similarity is not None and similarity > max_similarity:
                    max_similarity = similarity

    return max_similarity

def wordnet_similarity(query, doc1, doc2):
    # Tokenize the documents and query
    query_tokens = word_tokenize(query)
    doc1_tokens = word_tokenize(doc1)
    doc2_tokens = word_tokenize(doc2)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    query_tokens = [word for word in query_tokens if word.lower() not in stop_words]
    doc1_tokens = [word for word in doc1_tokens if word.lower() not in stop_words]
    doc2_tokens = [word for word in doc2_tokens if word.lower() not in stop_words]

    # Calculate WordNet similarity using Wu-Palmer similarity
    similarity_doc1 = calculate_wup_similarity(query_tokens, doc1_tokens)
    similarity_doc2 = calculate_wup_similarity(query_tokens, doc2_tokens)

    return similarity_doc1, similarity_doc2

def wup_similarity(synset1, synset2):
    return synset1.wup_similarity(synset2) if synset1 and synset2 else 0

def get_most_similar_term(query_terms):
    most_similar_term = None
    highest_similarity = 0

    for i, term1 in enumerate(query_terms):
        term1_synsets = wn.synsets(term1)
        avg_similarity = 0

        for j, term2 in enumerate(query_terms):
            if i != j:  # Avoid comparing a term to itself
                term2_synsets = wn.synsets(term2)
                max_pair_similarity = max(wup_similarity(s1, s2) for s1 in term1_synsets for s2 in term2_synsets)
                avg_similarity += max_pair_similarity

        avg_similarity /= len(query_terms) - 1
        if avg_similarity > highest_similarity:
            highest_similarity = avg_similarity
            most_similar_term = term1

    return most_similar_term
