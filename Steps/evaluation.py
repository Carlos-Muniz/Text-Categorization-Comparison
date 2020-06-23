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
                            zero_division=1),
            recall_score(test_y, results[r], average='micro', zero_division=1),
            f1_score(test_y, results[r], average='micro', zero_division=1),
            precision_score(test_y, results[r], average='macro',
                            zero_division=1),
            recall_score(test_y, results[r], average='macro', zero_division=1),
            f1_score(test_y, results[r], average='macro', zero_division=1)
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
    scores.to_csv('RESULTS/eval_results.csv')
    modvecs = scores['model'] + '\n' + scores['vectorization']
    ax = scores.plot.bar()
    plt.grid(b=True, axis='y', which='major')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
              ncol=4, fancybox=True, shadow=False)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(modvecs, rotation=45)
    plt.savefig('RESULTS/eval_results.png', bbox_inches='tight')
