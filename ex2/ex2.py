# imports:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

to_drop = ['date']
data_file_name = r'kc_house_data.csv'
# the data set
df = pd.read_csv(data_file_name)


def clean_data():
    """
    cleans the data from unused columns.
    :return: the prices of the houses by their id
    """
    df.drop_duplicates(subset=['id'], keep='first', inplace=True)
    df.set_index('id', inplace=True)
    year_of_sale = df['date'].str.extract(r'^(\d{4})', expand=False)
    df['year_of_sale'] = pd.to_numeric(year_of_sale)
    month_of_sale = df['date'].str.extract(r'^(\d{6})', expand=False)
    df['month_of_sale'] = pd.to_numeric(month_of_sale.str.replace(r'^(\d{4})', ''))
    df.drop(columns=to_drop, inplace=True, axis=1)
    df.drop([0], inplace=True, axis=0)
    df.dropna(inplace=True)
    for col in df.keys():
        if col != 'long':
            df[col] = abs(df[col])


def handle_categorical():
    """
    handles the categorial data and changes it to binary values
    :return:
    """
    one_hot = pd.get_dummies(df['zipcode'])
    df.drop('zipcode', inplace=True, axis=1)
    return one_hot


def feature_relation_graph(feature1, feature2):
    """
    plots a graph comparing the two features
    :param feature1:
    :param feature2:
    :return:
    """
    plt.scatter(df[feature1], df[feature2])
    plt.title(feature1 + " vs " + feature2)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()


def get_singular_values():
    """
    calculates the sungular values of the data matrix
    :return:
    """
    return np.linalg.svd(df, full_matrices=True, compute_uv=False)


def get_min_w(x_train, y_train):
    """
    calculates the min w for the samples
    :param x_train: samples
    :param y_train: real values (prices of the houses)
    :return: the min w
    """
    Xt_pseudo_inv = np.linalg.pinv(x_train)
    return Xt_pseudo_inv.dot(y_train.T)


def error_mse(X, w, y):
    """
    calculates the mse error
    :param X: the data
    :param w: the estimator
    :param y: the real results
    :return:
    """
    return np.average((y - X.dot(w)) ** 2)


def train_data_set():
    """
    trains the data and finds tha min w. does it 100 times while
    changing the sample size of the train comparing to the test
    :return: none
    """
    train_error = np.zeros(100)
    test_error = np.zeros(100)
    for x in range(1, 100):
        msk = np.random.rand(len(df)) < (x / 100)
        X_train = df[msk].iloc[:, 1:]
        X_test = df[~msk].iloc[:, 1:]
        Y_train = df[msk].iloc[:, 0]
        Y_test = df[~msk].iloc[:, 0]
        w = get_min_w(X_train, Y_train)
        train_error[x - 1] = error_mse(X_train, w, Y_train)
        test_error[x - 1] = error_mse(X_test, w, Y_test)
        # print("train error: ", train_error, "test error: ", test_error)

    return train_error, test_error


def get_correlation():
    """
    plots the corr relations between all the features
    :return:
    """
    plt.colorbar(plt.matshow(abs(df.corr())))
    plt.xticks(range(len(df.columns)), df.columns, rotation='vertical')
    plt.yticks(range(len(df.columns)), df.columns)

    plt.show()


def plot_errors(train_error, test_error):
    """
    plots both errors (test and train)
    :param train_error:
    :param test_error:
    :return: none
    """
    plt.plot(train_error, color="blue", label="train error")
    plt.plot(test_error, color="orange", label="test error")
    plt.xlabel("x- the size of the training set in % from the samples")
    plt.ylabel("error")
    plt.title("train and test errors as a function of the num of x")
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    clean_data()
    categorical = handle_categorical()
    df = df.join(categorical)
    # singular_values = get_singular_values()
    train_error, test_error = train_data_set()
    plot_errors(train_error, test_error)
