import numpy as np

def divide_on_feature(X, feature_i, threshold):

    mask = X[:, feature_i] >= threshold

    return [X[mask], X[~mask]]