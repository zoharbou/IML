from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

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
    x_train = [np.delete(line, -1) for line in train]
    y_test = np.array([line[-1] for line in test])
    x_test = [np.delete(line, -1) for line in test]
    return x_train, y_train, x_test, y_test


def train_round():
    x_train, y_train, x_test, y_test = get_test_train()
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    predict_prob = clf.predict_proba(x_test)
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
    plt.show()


def train_data(trails_n):
    fprs = []
    tprs = []
    for i in range(0, trails_n):
        fpr, tpr = train_round()
        fprs.append(fpr)
        tprs.append(tpr)
    plot_ROC(np.mean(pad_to_max(fprs), axis=0), np.mean(pad_to_max(tprs), axis=0))


if __name__ == '__main__':
    df = import_data("data\spambase.data")
    train_data(10)
