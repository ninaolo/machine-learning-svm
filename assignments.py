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
    return numpy.dot(x, y) + 1

