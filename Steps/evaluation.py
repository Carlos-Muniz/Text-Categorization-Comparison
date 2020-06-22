#!/usr/bin/env base
# encoding: utf-8

"""
AUTHOR(S):      Carlos M. Muniz
CREATION DATE:  2020-06-22
FILE NAME:      evaluation.py
DESCRIPTION:    This script evaluates the results.
"""
import pdb
import pickle
import numpy as np
import pandas as pd
from glob import glob
from os.path import basename
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def evaluate(results_path, test_y):
    with open(results_path, 'rb') as fin:
        results = pickle.load(fin)
    mname = basename(results_path).split('.')[0]
    scores = []
    for r in results:
        scores.append([
            mname, r, accuracy_score(test_y, results[r]),
            precision_score(test_y, results[r], average='micro',
                            zero_division=0),
            recall_score(test_y, results[r], average='micro', zero_division=0),
            f1_score(test_y, results[r], average='micro', zero_division=0),
            precision_score(test_y, results[r], average='macro',
                            zero_division=0),
            recall_score(test_y, results[r], average='macro', zero_division=0),
            f1_score(test_y, results[r], average='macro', zero_division=0)
        ])
    return scores


def evaluation():
    test_y = np.load('DATA/test_y.npy', allow_pickle=True)
    all_results = []
    for results in glob('RESULTS/*.pkl'):
        all_results.extend(evaluate(results, test_y))
    cols = [
        'model', 'vectorization', 'accuracy',
        'precision_micro', 'recall_micro', 'f1_micro',
        'precision_macro', 'recall_macro', 'f1_macro'
    ]
    scores = pd.DataFrame(all_results, columns=cols).sort_values('accuracy')
    modvecs = scores['model'] + '\n' + scores['vectorization']
    ax = scores.plot.bar()
    plt.grid(b=True, axis='y', which='major')
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(modvecs, rotation=0)
    plt.title('Evaluation Results')
    plt.savefig('RESULTS/eval_results.png', bbox_inches='tight')
    # pdb.set_trace()
