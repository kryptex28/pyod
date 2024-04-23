# -*- coding: utf-8 -*-
"""Example of using Histogram- based outlier detection (HBOS) for
outlier detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys
from sklearn.metrics import roc_auc_score
# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.hbospyod import HBOSPYOD
from pyod.models.hbos import HBOS
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from pyod.utils.utility import precision_n_scores


if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

    # Generate sample data
    X_train, X_test, y_train, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=2,
                      contamination=contamination,
                      random_state=42)

    # train HBOS detector
    clf_name = 'HBOSPYOD'
    clf_name2= 'HBOS'
    clf = HBOSPYOD()
    clf.fit(X_train)
    clf2= HBOS()
    clf2.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_pred2 = clf2.labels_  # binary labels (0: inliers, 1: outliers)

    y_train_scores = clf.decision_scores_  # raw outlier scores
    y_train_scores2 = clf.decision_scores_

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    y_test_pred2 = clf2.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores2 = clf2.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)

    print("\nOn Training Data2:")
    evaluate_print(clf_name2, y_train, y_train_scores2)
    print("\nOn Test Data2:")
    evaluate_print(clf_name2, y_test, y_test_scores2)

    print(clf.decision_scores_,"clf")
    print(clf2.decision_scores_,"clf2")

    # visualize the results
    #visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
     #         y_test_pred, show_figure=True, save_figure=False)

    #visualize(clf_name2, X_train, y_train, X_test, y_test, y_train_pred2,
     #         y_test_pred2, show_figure=True, save_figure=False)

    print(roc_auc_score(y_test, y_test_scores),"roc_auc_score")
    print(roc_auc_score(y_test, y_test_scores2),"roc_auc_score2")
    clf3=HBOSPYOD()

    print(clf3.fit_predict_score( X_test, y_test, scoring='prc_n_score')," prc_n_score")
    clf3.fit(X_train)
    print(clf3.decision_scores_)
    print(clf3.decision_function(X_train))
    print(precision_n_scores(y_test,y_test_scores))