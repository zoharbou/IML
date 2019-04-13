import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

mean = [0, 0, 0]
scaling_matrix = np.diag([0.1, 0.5, 2])


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


if __name__ == '__main__':
    # ex 23
    cov = np.eye(3)
    x_y_z = np.random.multivariate_normal(mean, cov, 50000).T
    plot_3d(x_y_z)
    plt.show()

    # ex 24
    cov = np.matmul((np.matmul(scaling_matrix, cov)), np.transpose(scaling_matrix))
    x_y_z = np.random.multivariate_normal(mean, cov, 50000).T
    plot_3d(x_y_z)
    plt.show()
    print("cov matrix analytic calculation:")
    print(cov)
    print()
    print("cov matrix numeric calculation:")
    print(np.matmul(x_y_z, x_y_z.transpose()) / 49999)
    print()

    # ex 25
    ort_matrix = get_orthogonal_matrix(3)
    cov = np.matmul((np.matmul(ort_matrix, cov)), np.transpose(ort_matrix))
    x_y_z = np.random.multivariate_normal(mean, cov, 50000).T
    plot_3d(x_y_z)
    plt.show()
    print("cov matrix analytic calculation:")
    print(cov)
    print()
    print("cov matrix numeric calculation:")
    print(np.matmul(x_y_z, x_y_z.transpose()) / 49999)
    print()

    # ex 26
    projection_matrix = np.diag([1, 1, 0])
    plot_2d(np.matmul(projection_matrix, x_y_z))
    plt.show()

    # ex 27
    new_x_y_x = np.array([list for list in x_y_z.transpose() if 0.1 > list[2] > (-0.4)]).transpose()
    plot_2d(np.matmul(projection_matrix, new_x_y_x))
    plt.show()
