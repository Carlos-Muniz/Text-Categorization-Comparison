#!/usr/bin/env base
# encoding: utf-8

"""
AUTHOR(S):      Carlos M. Muniz
CREATION DATE:  2020-06-18
FILE NAME:      preprocess.py
DESCRIPTION:    This script contains code that preprocesses the data
"""
import pdb
import pickle
import string
import numpy as np
import pandas as pd
from nltk import data
from nltk import download
from nltk import word_tokenize
from nltk.corpus import reuters
from collections import Counter
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Loads the raw text and labels from the Reuters Corpus in to separate
# Training and Testing Dataframes
def load_data():
    docs = reuters.fileids()
    train_ids = [doc for doc in docs if doc.startswith("train")]
    test_ids = [doc for doc in docs if doc.startswith("test")]

    train_data = pd.DataFrame(
        [(reuters.raw(id), reuters.categories(id)[0]) for id in train_ids],
        columns=('text', 'labels'))

    test_data = pd.DataFrame(
        [(reuters.raw(id), reuters.categories(id)[0]) for id in test_ids],
        columns=('text', 'labels'))

    return train_data, test_data


# Token Analysis to help define the length of a vector
def token_analysis(train, test):
    stop = stopwords.words('english') + list(string.punctuation)
    train_sents = train['text'].values
    train_words = []
    for sent in train_sents:
        for w in word_tokenize(sent.lower()):
            if w not in stop:
                train_words.append(w)
    count = Counter(train_words)
    cutoffs = list(range(1, 251))
    wordcount = [len([x for x, y in count.items() if y >= i]) for i in cutoffs]
    fig = plt.figure()
    plt.plot(cutoffs, wordcount)
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Words that appear atleast n times')
    plt.ylabel('Number of unique words')
    plt.title("Zipf\'s law")
    plt.savefig('DATA/corpus_count.png', bbox_inches='tight')
    # pdb.set_trace()


# Vectorizes and saves the training and testing data using Bag of Words
def bagofwords_vectorization(train, test, n=1000):
    vectorizer = CountVectorizer(stop_words='english', max_features=n)
    # Vectorization
    train_X = vectorizer.fit_transform(train['text'].values).toarray()
    test_X = vectorizer.transform(test['text'].values).toarray()
    # Save vectors
    np.save('DATA/train_bagofwords_{0}_X'.format(n), train_X)
    np.save('DATA/test_bagofwords_{0}_X'.format(n), test_X)


# Vectorizes and saves the training and testing data using TFIDF
def tfidf_vectorization(train, test, n=1000):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=n)
    # Vectorization
    train_X = vectorizer.fit_transform(train['text'].values).toarray()
    test_X = vectorizer.transform(test['text'].values).toarray()
    # Save vectors
    np.save('DATA/train_tfidf_{0}_X'.format(n), train_X)
    np.save('DATA/test_tfidf_{0}_X'.format(n), test_X)


# Maps labels for classification and writes to file
def labels_mapping(train, test):
    n = len(train['labels'].values)
    labels = np.hstack([train['labels'].values, test['labels'].values])
    # Map Labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    classes = {i: c for i, c in enumerate(le.classes_)}
    train_y = encoded_labels[:n]
    test_y = encoded_labels[n:]
    # Write to file
    with open('DATA/label_classes.pkl', 'wb') as fout:
        pickle.dump(classes, fout)
    np.save('DATA/train_y', train_y)
    np.save('DATA/test_y', test_y)


def preprocess():
    # Load Raw Data
    train, test = load_data()
    # Token Analysis
    token_analysis(train, test)
    # Encode labels
    labels_mapping(train, test)
    for num in [500, 1000, 2000]:
        # Bag of Words
        bagofwords_vectorization(train, test, n=num)
        # TFIDF
        tfidf_vectorization(train, test, n=num)
