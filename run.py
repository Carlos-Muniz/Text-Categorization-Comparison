#!/usr/bin/env base
# encoding: utf-8

"""
AUTHOR(S):      Carlos M. Muniz
CREATION DATE:  2020-06-18
FILE NAME:      run.py
DESCRIPTION:    This script runs the Text-Categorization Process
"""
from Steps.preprocess import preprocess
from Steps.models import train
from Steps.models import test
from Steps.evaluation import evaluation


def main():
    # Step 1: Load and Preprocess the data into vectors
    preprocess()
    # Step 2: Train and test models
    train()
    test()
    # Step 3: Evaluate the performance of the models
    evaluation()
    pass


if __name__ == '__main__':
    main()
