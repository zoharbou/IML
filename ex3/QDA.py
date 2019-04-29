import numpy as np


class QDA:
    def __init__(self):
        # self.x_train = None
        # self.y_train = None
        self.mu_estimation = {}
        self.N_y = {}
        self.classes = None

    def fit(self, X, y):
        # self.x_train = X
        # self.y_train = y
        self.classes = set(y)
        print(self.classes)
        for cl in self.classes:
            class_x = X[y == cl]
            self.N_y[cl] = len(class_x)
            self.mu_estimation[cl] = np.mean(class_x, axis=0)
        print(self.mu_estimation)
        print(self.N_y)

    def predict(self, x):
        return
