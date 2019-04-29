import numpy as np
from scipy.spatial import distance


class LDA:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.mu_estimation = {}
        self.N_y = {}
        self.classes = None
        self.d = None
        self.cov_matrix = None
        self.global_cov_matrix = None
        self.pi_y = {}

    def get_mu_est(self):
        for cl in self.classes:
            class_x = self.x_train[self.y_train == cl]
            self.N_y[cl] = len(class_x)
            self.mu_estimation[cl] = np.mean(class_x, axis=0)

    def get_cov_matrix(self):
        S = np.zeros((self.d, self.d))
        for cl in self.classes:
            mv = self.mu_estimation[cl]
            # S matrix for every class
            class_s_mat = np.zeros((self.d, self.d))
            for row in self.x_train[self.y_train == cl]:
                row, mv = row.reshape(self.d, 1), mv.reshape(self.d, 1)
                class_s_mat += (row - mv).dot((row - mv).T)
            S += class_s_mat * (self.N_y[cl] - 1)
        return S / (sum(self.N_y.values()) - len(self.classes))

    def get_global_cov(self):
        overall_mean = np.mean(self.x_train, axis=0)
        global_cov = np.zeros((self.d, self.d))
        for i in range(len(self.mu_estimation)):
            n = self.x_train[self.y_train == i + 1, :].shape[0]
            mean_vec = self.mu_estimation[i].reshape(self.d, 1)  # make column vector
            overall_mean = overall_mean.reshape(self.d, 1)  # make column vector
            global_cov += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        return global_cov

    def fit(self, X, y):
        self.x_train = X
        self.y_train = y
        self.d = len(self.x_train[0])
        self.classes = set(y)
        self.get_mu_est()
        self.cov_matrix = self.get_cov_matrix()
        self.global_cov_matrix = self.get_global_cov()
        for cl, N in self.N_y.items():
            self.pi_y[cl] = N / len(self.x_train)

    def prob_y(self, x, y):
        # return x.T * np.linalg.inv(self.cov_matrix) * self.mu_estimation[y] - 0.5 * self.mu_estimation[
        #     y].T * np.linalg.inv(self.cov_matrix) * self.mu_estimation[y] + np.log(self.pi_y[y])
        return np.matmul(self.mu_estimation[y], self.cov_matrix).dot(x) - (
                    np.matmul(self.mu_estimation[y], self.cov_matrix).dot(self.mu_estimation[y]) * 0.5) + np.log(
            self.pi_y[y])

    def predict(self, x):
        predictions = {}
        for cl in self.classes:
            predictions[int(cl)] = self.prob_y(x, cl)
        return max(predictions, key=predictions.get)
