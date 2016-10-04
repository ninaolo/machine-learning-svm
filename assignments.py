# Using cvxopt for solving convex optimization problems
# qp = quadratic optimization problem solver
# matrix = takes matrix and converts to cvxopt matrix to be passed to qp

# The call to qp can look like this:
# r = qp(matrix(P) , matrix(q) , matrix(G) , matrix(h))
# alpha = list(r[’x’])

# The above call finds the alpha which minimizes the below:
# 0.5 * alpha_T * matrix(P) * alpha + matrix(q)_T * alpha         ( while matrix(G) * alpha <= matrix(h) )

from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, pylab, random, math


def linear_kernel(x, y):
    """ Returns the linear kernel function of two vectors x and y. """
    return numpy.dot(x, y) + 1


def create_random_classified_test_data():
    """ Creates random data points (x, y) with two classes -1 and 1. """

    class_a = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + [
        (random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)]

    class_b = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]

    data = class_a + class_b
    random.shuffle(data)

    return data


def plot_data(class_a, class_b):
    """ Plots two classes of data. """
    pylab.hold(True)

    pylab.plot([p[0] for p in class_a], [p[1] for p in class_a], 'bo')
    pylab.plot([p[0] for p in class_b], [p[1] for p in class_b], 'ro')

    pylab.show()


def create_p_matrix(data, kernel_function):
    """ Creates an N x N matrix P.
    P_ij = t_i * t_j * K(x_i, x_j)
    N is the number of data points.
    K is a kernel function.
    t is the class (-1 or 1).
    x is a vector with data points.
    """

    N = len(data)
    P = numpy.zeros(N, N)

    for i in range(N):
        for j in range(N):
            t_i = data[i][2]
            t_j = data[j][2]
            x_i = data[i][:2]
            x_j = data[j][:2]

            P[i, j] = t_i * t_j * kernel_function(x_i, x_j)

    return P






