import numpy as np
from scipy.spatial import distance


class knn:
    def __init__(self, k):
        self.__k = k
        self.__X_train = None
        self.__y_train = None

    def fit(self, X, y):
        self.__X_train = X
        self.__y_train = y

    def predict(self, x):
        dist = np.zeros(len(self.__X_train))
        for i, sample in enumerate(self.__X_train):
            dist[i] = distance.euclidean(x, sample)
        sorted_y = self.__y_train[np.argsort(dist, kind='mergesort')]
        (values, counts) = np.unique(sorted_y[:self.__k], return_counts=True)
        return values[np.argmax(counts)]


