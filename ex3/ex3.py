from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

t = [0.2, 0.4, 0.55, 0.95]


def get_lin_space(mu, sigma):
    return np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)


def plot_pdf_normal(mu, sigma, color):
    x = get_lin_space(mu, sigma)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=5, alpha=0.6,
             label='mu= ' + str(mu) + ' sigma = ' + str(sigma), color=color)


def plot_cdf_normal(mu, sigma, color):
    x = get_lin_space(mu, sigma)
    plt.plot(x, norm.cdf(x, mu, sigma), 'r-', lw=5, alpha=0.6,
             label='mu= ' + str(mu) + ' sigma = ' + str(sigma), color=color)


def f(x):
    return sp.expit(x)


def h(values):
    answer = np.zeros(len(values))
    i = 0
    w = np.array([10, -10])
    for x in values:
        answer[i] = f((np.append(np.array(x), 1)).dot(w))
        i += 1
    return answer


def plot_h():
    x = np.linspace(-1.5, 3, 1000)
    plt.plot(x, h(x), color='purple')
    plt.title("the hypothesis h(x) as a function of x")
    plt.show()


def question5b():
    plot_pdf_normal(-4, 1, "red")
    plot_pdf_normal(6, 1, "blue")
    plt.legend(loc='upper center')
    plt.title("norm probability density function:")
    plt.show()

    plot_cdf_normal(-4, 1, "red")
    plot_cdf_normal(6, 1, "blue")
    plt.legend(loc='center')
    plt.title("norm cumulative distribution function:")
    plt.show()


def hx_on_normal(mu, sigma):
    x = np.linspace(0, 1, 1000)
    plt.plot(x, norm.ppf(h(x), mu, sigma), label='mu= ' + str(mu) + ' sigma = ' + str(sigma))
    plt.title("h(x) cumulative distribution function:\n"
              "mu = " + str(mu) + " sigma = " + str(sigma))
    plt.show()


def plot_cdf_funcs_hx():
    hx_on_normal(-4, 1)
    hx_on_normal(6, 1)


if __name__ == '__main__':
    # question5b()
    plot_h()
    plot_cdf_funcs_hx()
