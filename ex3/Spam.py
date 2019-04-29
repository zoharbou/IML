from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from knn import knn
from QDA import QDA
from LDA import LDA

import matplotlib.pyplot as plt


def clean(line):
    return np.array(list(map(float, line.rstrip('\n').split(","))))


def import_data(file_path):
    return [clean(line) for line in open(file_path)]


def compare(item):
    return item.size


def get_test_train():
    train, test = train_test_split(df, test_size=1000)
    y_train = np.array([line[-1] for line in train])
    x_train = np.array([np.delete(line, -1) for line in train])
    y_test = np.array([line[-1] for line in test])
    x_test = np.array([np.delete(line, -1) for line in test])
    return x_train, y_train, x_test, y_test


def train_logistic_reg(x_train, y_train, x_test, proba=False):
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    if proba:
        return clf.predict_proba(x_test)
    return clf.predict(x_test)


def train_round():
    x_train, y_train, x_test, y_test = get_test_train()
    predict_prob = train_logistic_reg(x_train, y_train, x_test, True)
    y_score = np.array([row[1] for row in predict_prob])
    sorted_index = np.argsort(y_score, kind="mergesort")[::-1]
    y_test = y_test[sorted_index]

    NP = int(np.sum(y_test))
    NN = int(y_test.size - NP)
    TPR = np.zeros(NP + 1)
    FPR = np.zeros(NP + 1)
    tp = np.cumsum(y_test)
    for i in range(1, NP + 1):
        TPR[i] = i / NP
        Ni = np.abs(tp - i).argmin() + 1
        FPR[i] = (Ni - i) / NN
    np.append(TPR, 1)
    np.append(FPR, 1)
    return FPR, TPR


def pad_to_max(array):
    max_size = sorted(array, key=compare)[-1].size
    for i in range(len(array)):
        to_add = max_size - array[i].size
        if to_add > 0:
            array[i] = np.append(array[i], [1] * to_add)
    return array


def plot_ROC(x, y):
    plt.plot(np.append(x, 1), np.append(y, 1), color="C6")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.title("ROC curve")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.show()


def log_reg_mean_trails(trails_n):
    fprs = []
    tprs = []
    for i in range(0, trails_n):
        fpr, tpr = train_round()
        fprs.append(fpr)
        tprs.append(tpr)
    plot_ROC(np.mean(pad_to_max(fprs), axis=0), np.mean(pad_to_max(tprs), axis=0))


def error_rate(y_prediction, y_real):
    return len(np.where(y_prediction != y_real)[0]) / len(y_real)


def compare_train_methods(neighbors_num):
    x_train, y_train, x_test, y_test = get_test_train()
    errors = {}
    for i, k in enumerate(neighbors_num):
        y_predict = train_knn(x_train, y_train, x_test, k)
        errors["knn k = " + str(k)] = error_rate(y_predict, y_test)
    y_predict = train_logistic_reg(x_train, y_train, x_test)
    errors["logistic reg: "] = error_rate(y_predict, y_test)
    print('error rates for the different train methods:')
    for method, error in errors.items():
        print('{} {}'.format(method, error))


def train_knn(x_train, y_train, x_test, k):
    trainer = knn(k)
    trainer.fit(x_train, y_train)
    y_predict = np.zeros(len(x_test))
    for j, x in enumerate(x_test):
        y_predict[j] = trainer.predict(x)
    return y_predict


def train_QDA():
    x_train, y_train, x_test, y_test = get_test_train()
    qda = QDA()
    qda.fit(x_train, y_train)


def train_LDA():
    x_train, y_train, x_test, y_test = get_test_train()
    lda = LDA()
    lda.fit(x_train, y_train)
    y_prediction = np.zeros(len(y_test))
    for i, x in enumerate(x_test):
        y_prediction[i] = lda.predict(x)
    print(y_prediction)
    print(error_rate(y_prediction, y_test))
    y_train_pred = np.zeros(len(y_train))
    for i, x in enumerate(x_train):
        y_train_pred[i] = lda.predict(x)
    print(error_rate(y_train_pred, y_train))


if __name__ == '__main__':
    df = import_data("data\spambase.data")
    # log_reg_mean_trails(10)
    # compare_train_methods([1, 2, 5, 10, 100])
    train_LDA()
