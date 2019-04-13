import numpy
import matplotlib.pyplot as plt


data = numpy.random.binomial(1, 0.25, (100000, 1000))
epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
NUM_OF_LINES = 100000
NUM_OF_TOSSES = 1000
P = 0.25


def f(m, k):
    """
    this function calculates the average of the m first numbers in the k line of the data
    :param m: the number of tosses
    :param k: the line of data
    :return: a list of the results
    """
    return list(map(lambda x: numpy.average(data[k][:x]), m))


def g(eps):
    """
    this function calculates the percentage of sequences that satisfy the distance from epsilon
    :param eps: the given max distance
    :return: a list of the results
    """
    ans = numpy.zeros(NUM_OF_TOSSES)
    sums = numpy.zeros(NUM_OF_LINES)

    for m in range(NUM_OF_TOSSES):
        sums += data[:, m]
        diff = numpy.fabs((sums / (m + 1)) - P)
        ans[m] += numpy.count_nonzero(diff > eps)
    return ans / NUM_OF_LINES


def chebyshev(m, eps):
    """
    calculates the upper bound by chebyshev for the given epsilon
    :param m: the number of tosses
    :param eps: the given epsilon
    :return: a list of the results
    """
    return 1 / (m * 4.0 * eps ** 2)


def hoeffding(m, eps):
    """
    calculates the upper bound by hoeffding for the given epsilon
    :param m: the number of tosses
    :param eps: the given epsilon
    :return: a list of the results
    """
    return 2 / numpy.exp(2.0 * m * eps ** 2)


def make_plot():
    """
    this function crates the plot for question num 29. it presents the mean of all
    tosses up to m for every m from 1 to 1000
    :return: none
    """
    t = numpy.arange(1, 1000, 1)
    plt.plot(f(t, 0), color="red", label="line number 1")
    plt.plot(f(t, 1), color="blue", label="line number 2")
    plt.plot(f(t, 2), color="green", label="line number 3")
    plt.plot(f(t, 3), color="orange", label="line number 4")
    plt.plot(f(t, 4), color="purple", label="line number 5")
    plt.ylabel('the mean of all tosses up to m')
    plt.xlabel("number of tosses (m)")
    plt.title("question number 29")
    plt.legend(loc='upper right')
    plt.show()


def make_bound_plot():
    """
    this function crates the plot that presents for every epsilon
    (from the list above) and for every m, what is the upper bound of the
    probability to get a mean that is far from the expected value more then epsilon, by hoeffding and chebyshev.
    it also presents the percentage of sequences that satisfy the distance from epsilon
    :return: none
    """
    t = numpy.arange(1, 1000, 1)
    i = 0
    for e in epsilon:
        plt.figure(i)
        i += 1
        plt.plot(numpy.clip(chebyshev(t, e), 0, 1), color="red", label="Chebyshev")
        plt.plot(numpy.clip(hoeffding(t, e), 0, 1), color="blue", label="Hoeffding")
        plt.plot(g(e), color="green", label="percentage of sequences that\nsatisfy the distance from epsilon")
        plt.xlabel("number of tosses (m)")
        plt.title("epsilon = " + str(e))
        plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    make_plot()
    make_bound_plot()
