import json
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

def eval_randomsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn classifier and a parameter grid to search,
    choose the optimal parameters from pgrid using Random Search CV
    and train the model using the training dataset and evaluate the
    performance on the test dataset. The random search cv should try
    at most 33% of the possible combinations.

    Parameters
    ----------
    clf : sklearn.ClassifierMixin
        The sklearn classifier model 
    pgrid : dict
        The dictionary of parameters to tune for in the model
    xTrain : nd-array with shape (n, d)
        Training data
    yTrain : 1d array with shape (n, )
        Array of labels associated with training data
    xTest : nd-array with shape (m, d)
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    resultDict: dict
        A Python dictionary with the following 4 keys,
        "AUC", "AUPRC", "F1", "Time" and the values are the floats
        associated with them for the test set.
    roc : dict
        A Python dictionary with 2 keys, fpr, and tpr, where
        each of the values are lists of the fpr and tpr associated
        with different thresholds. You should be able to use this
        to plot the ROC for the model performance on the test curve.
    bestParams: dict
        A Python dictionary with the best parameters chosen by your
        GridSearch. The values in the parameters should be something
        that was in the original pgrid.
    """
    permutations = np.prod([len(v) for v in pgrid.values()])
    start = time.time()

    cv = RandomizedSearchCV(clf, param_distributions=pgrid, n_iter=int(permutations*0.33), cv=10)
    cv.fit(xTrain, yTrain)

    timeElapsed = time.time() - start

    clf = cv.best_estimator_
    best_params = cv.best_params_

    yHat = cv.predict(xTest)
    yHat_proba = cv.predict_proba(xTest)

    auc = roc_auc_score(yTest, yHat_proba, multi_class='ovr')

    auprc = average_precision_score(yTest, yHat_proba)

    f1 = f1_score(yTest, yHat, average='weighted')

    return {'AUC': auc, 'AUPRC': auprc, 'F1': f1, 'Time': timeElapsed}, best_params


def eval_searchcv(clfName, clf, clfGrid,
                  xTrain, yTrain, xTest, yTest,
                  perfDict, bestParamDict):
    # evaluate random search and add to perfDict
    clfr_perf, rs_p  = eval_randomsearch(clf, clfGrid, xTrain,
                                            yTrain, xTest, yTest)
    perfDict[clfName + " (Random)"] = clfr_perf
    bestParamDict[clfName] = {"Random": rs_p}
    return perfDict, bestParamDict


def your_model():
    return LogisticRegression(C=10, tol=0.0004, penalty='l1', solver='liblinear')



def get_parameter_grid(mName):
    """
    Given a model name, return the parameter grid associated with it

    Parameters
    ----------
    mName : string
        name of the model (e.g., DT, KNN, LR (None))

    Returns
    -------
    pGrid: dict
        A Python dictionary with the appropriate parameters for the model.
        The dictionary should have at least 2 keys and each key should have
        at least 2 values to try.
    """
    if mName == 'DT':
        return {'max_depth': [17, 20, 25], 'min_samples_leaf': [13, 15, 18]}
    elif mName == 'LR (None)':
        return {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'tol': [0.0001, 0.0004]}
    elif mName == 'LR (L1)':
        return {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'tol': [0.0001, 0.0004]}
    elif mName == 'LR (L2)':
        return {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'tol': [0.0001, 0.0004]}
    elif mName == 'KNN':
        return {'n_neighbors': [5, 10, 20, 40], 'p': [1, 2]}
    elif mName == 'NN':
        return {'alpha':[0.0001, 0.0005, 0.001], 'hidden_layer_sizes': [(50, 50), (100, 100)]}


def main():
    # load the train and test data
    xTrain = pd.read_csv("xTrain.csv").to_numpy()
    yTrain = pd.read_csv("yTrain.csv").to_numpy().flatten()
    xTest = pd.read_csv("xTest.csv").to_numpy()
    yTest = pd.read_csv("yTest.csv").to_numpy().flatten()

    perfDict = {}
    bestParamDict = {}

    print("Tuning Decision Tree --------")
    # Compare Decision Tree
    dtName = "DT"
    dtGrid = get_parameter_grid(dtName)
    # fill in
    dtClf = DecisionTreeClassifier()
    perfDict, bestParamDict = eval_searchcv(dtName, dtClf, dtGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, bestParamDict)
    print("Tuning Unregularized Logistic Regression --------")
    # logistic regression (unregularized)
    unregLrName = "LR (None)"
    unregLrGrid = get_parameter_grid(unregLrName)
    # fill in
    lrClf = LogisticRegression(max_iter=500)
    perfDict, bestParamDict = eval_searchcv(unregLrName, lrClf, unregLrGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, bestParamDict)
    # logistic regression (L1)
    print("Tuning Logistic Regression (Lasso) --------")
    lassoLrName = "LR (L1)"
    lassoLrGrid = get_parameter_grid(lassoLrName)
    # fill in
    lassoClf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
    perfDict, bestParamDict = eval_searchcv(lassoLrName, lassoClf, lassoLrGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, bestParamDict)
    # Logistic regression (L2)
    print("Tuning Logistic Regression (Ridge) --------")
    ridgeLrName = "LR (L2)"
    ridgeLrGrid = get_parameter_grid(ridgeLrName)
    # fill in
    ridgeClf = LogisticRegression(penalty='l2', max_iter=500)
    perfDict, bestParamDict = eval_searchcv(ridgeLrName, ridgeClf, ridgeLrGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, bestParamDict)
    # k-nearest neighbors
    print("Tuning K-nearest neighbors --------")
    knnName = "KNN"
    knnGrid = get_parameter_grid(knnName)
    # fill in
    knnClf = KNeighborsClassifier()
    perfDict, bestParamDict = eval_searchcv(knnName, knnClf, knnGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, bestParamDict)
    # neural networks
    # print("Tuning neural networks --------")
    # nnName = "NN"
    # nnGrid = get_parameter_grid(nnName)
    # # fill in
    # nnClf = MLPClassifier(max_iter=300)
    # perfDict, bestParamDict = eval_searchcv(nnName, nnClf, nnGrid,
    #                                                xTrain, yTrain, xTest, yTest,
    #                                                perfDict, bestParamDict)
    perfDF = pd.DataFrame.from_dict(perfDict, orient='index')
    print(perfDF)
    # store the best parameters
    with open('parameters', 'w') as f:
        json.dump(bestParamDict, f)


if __name__ == "__main__":
    main()
