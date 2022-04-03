# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

from sys import path
from os import path as ospath
path.append(f'{ospath.dirname(__file__)}/..')
import numpy as np
from board_edge_detection.coordinates_system import Line
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import preprocessing

def find_lines(grads, y_intercepts, x_intercepts):
    X = np.array([grads, y_intercepts, x_intercepts])
    X = np.reshape(X, (X.shape[1], X.shape[0]))
    print(X.shape)
    clusters = SpectralClustering(4, random_state=0).fit(X)
    labels = clusters.labels_
    return labels
    
def manual_classification(grads, y_intercepts, x_intercepts):
    horizontal_grad_bounds = [-0.05, 0, 0.05]
    vertical_grad_bounds = []

def normalise_infs_nans(array, max, min):
    array = np.array(array)
    for i, item in enumerate(array):
        if np.isinf(item):
            array[i] = max
        if np.isnan(item):
            array[i] = max
    
    norm = array / max
    return norm


if __name__ == "__main__":
    find_lines([], [], [])