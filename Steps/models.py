#!/usr/bin/env base
# encoding: utf-8

"""
AUTHOR(S):      Carlos M. Muniz
CREATION DATE:  2020-06-22
FILE NAME:      models.py
DESCRIPTION:    This script contains the functions for training and testing
                models
"""
import pdb
import pickle
import numpy as np
from glob import glob
from os.path import basename
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier


def logreg_models(x_paths, train_y):
    models = {}
    for f in x_paths:
        fname = "_".join(f.split('_')[1:3])
        train_x = np.load(f, allow_pickle=True)
        models[fname] = LogisticRegression(random_state=1, max_iter=1000)
        models[fname].fit(train_x, train_y)
    with open('MODELS/sklearn_logreg.pkl', 'wb') as fout:
        pickle.dump(models, fout)


def linsvc_models(x_paths, train_y):
    models = {}
    for f in x_paths:
        fname = "_".join(f.split('_')[1:3])
        train_x = np.load(f, allow_pickle=True)
        models[fname] = LinearSVC(random_state=1)
        models[fname].fit(train_x, train_y)
    with open('MODELS/sklearn_linsvc.pkl', 'wb') as fout:
        pickle.dump(models, fout)


def pasagg_models(x_paths, train_y):
    models = {}
    for f in x_paths:
        fname = "_".join(f.split('_')[1:3])
        train_x = np.load(f, allow_pickle=True)
        models[fname] = PassiveAggressiveClassifier(random_state=1)
        models[fname].fit(train_x, train_y)
    with open('MODELS/sklearn_pasagg.pkl', 'wb') as fout:
        pickle.dump(models, fout)


def sgdclf_models(x_paths, train_y):
    models = {}
    for f in x_paths:
        fname = "_".join(f.split('_')[1:3])
        train_x = np.load(f, allow_pickle=True)
        models[fname] = SGDClassifier(random_state=1, penalty='elasticnet')
        models[fname].fit(train_x, train_y)
    with open('MODELS/sklearn_sgdclf.pkl', 'wb') as fout:
        pickle.dump(models, fout)


def ranfor_models(x_paths, train_y):
    models = {}
    for f in x_paths:
        fname = "_".join(f.split('_')[1:3])
        train_x = np.load(f, allow_pickle=True)
        models[fname] = RandomForestClassifier(random_state=1)
        models[fname].fit(train_x, train_y)
    with open('MODELS/sklearn_ranfor.pkl', 'wb') as fout:
        pickle.dump(models, fout)


def percep_models(x_paths, train_y):
    models = {}
    for f in x_paths:
        fname = "_".join(f.split('_')[1:3])
        train_x = np.load(f, allow_pickle=True)
        models[fname] = Perceptron(random_state=1)
        models[fname].fit(train_x, train_y)
    with open('MODELS/sklearn_percep.pkl', 'wb') as fout:
        pickle.dump(models, fout)


def knnclf_models(x_paths, train_y):
    models = {}
    for f in x_paths:
        fname = "_".join(f.split('_')[1:3])
        train_x = np.load(f, allow_pickle=True)
        models[fname] = KNeighborsClassifier(n_neighbors=10)
        models[fname].fit(train_x, train_y)
    with open('MODELS/sklearn_knnclf.pkl', 'wb') as fout:
        pickle.dump(models, fout)


def mlpclf_models(x_paths, train_y):
    models = {}
    for f in x_paths:
        a, b = f.split('_')[1:3]
        fname = "_".join([a, b])
        train_x = np.load(f, allow_pickle=True)
        lyrs = (int(b) // 2, np.unique(train_y).shape[0])
        models[fname] = MLPClassifier(random_state=1, hidden_layer_sizes=lyrs)
        models[fname].fit(train_x, train_y)
    with open('MODELS/sklearn_mlpclf.pkl', 'wb') as fout:
        pickle.dump(models, fout)


def train():
    train_x_paths = glob('DATA/train_*_X.npy')
    train_y = np.load('DATA/train_y.npy', allow_pickle=True)
    # Logistic Regression Classification
    logreg_models(train_x_paths, train_y)
    # Linear SVC
    linsvc_models(train_x_paths, train_y)
    # Passive Agressive Classifier
    pasagg_models(train_x_paths, train_y)
    # SGD Classifier with Elasticnet penalty
    sgdclf_models(train_x_paths, train_y)
    # Random Forest Classifier
    ranfor_models(train_x_paths, train_y)
    # Perceptron
    percep_models(train_x_paths, train_y)
    # K-Nearest Neighbors
    knnclf_models(train_x_paths, train_y)
    # Multi-Layer Perceptron Classifier
    mlpclf_models(train_x_paths, train_y)


def test_models(x_paths, model_path):
    results = {}
    with open(model_path, 'rb') as fin:
        models = pickle.load(fin)
    for f in x_paths:
        fname = "_".join(f.split('_')[1:3])
        test_x = np.load(f, allow_pickle=True)
        results[fname] = models[fname].predict(test_x)
    results_path = 'RESULTS/' + basename(model_path)
    with open(results_path, 'wb') as fout:
        pickle.dump(results, fout)


def test():
    test_x_paths = glob('DATA/test_*_X.npy')

    for m in glob('MODELS/sklearn_*.pkl'):
        test_models(test_x_paths, m)
