# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd


def crossval(X, Y, n_iterations, iteration):

    len_X = len(X)

    i_start = int(iteration * (len_X / n_iterations))
    i_end = int((iteration + 1) * (len_X / n_iterations) - 1)

    i_test = np.arange(i_start, i_end + 1)
    i_app = np.setdiff1d(np.arange(len_X), i_test)

    return X[i_app], Y[i_app], X[i_test], Y[i_test]


def crossval_strat(X, Y, n_iterations, iteration):

    classes = np.unique(Y)
    n_classes = len(classes)
    len_X_c = len(X) // n_classes

    i_start = int(iteration * (len_X_c / n_iterations))
    i_end = int((iteration + 1) * (len_X_c / n_iterations) - 1)

    i_t_range = np.arange(i_start, i_end + 1)

    i_test = np.array([], dtype=int)

    for c in classes:
        filtered_i = np.where(Y == c)[0]
        i_test = np.append(i_test, filtered_i[i_t_range])

    i_app = np.setdiff1d(np.arange(len(X)), i_test)

    return X[i_app], Y[i_app], X[i_test], Y[i_test]


def analyse_perfs(numbers):
    return (np.mean(numbers), np.var(numbers))
