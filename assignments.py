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
    x_t = numpy.transpose(x)
    return numpy.dot(x_t, y) + 1


def polynomial_kernel(x, y):
    """ Returns the polynomial kernel function of two vectors x and y. """
    exp = 4
    x_t = numpy.transpose(x)
    return (numpy.dot(x_t, y) + 1) ** exp


def radial_basis_kernel(x, y):
    """ Returns the radial basis kernel function of two vectors x and y. """
    param = 5
    squared_euclidean_distance = 0
    for square in numpy.power(numpy.subtract(x, y), 2):
        squared_euclidean_distance += square

    return math.exp(-squared_euclidean_distance / (math.pow(2 * param, 2)))


def create_random_classified_test_data():
    """ Creates random data points (x, y) with two classes -1 and 1. """

    class_a = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + [
        (random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)]

    class_b = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]

    return class_a, class_b


def plot_data_points(class_a, class_b, indicator_list, kernel_function):
    """ Plots two classes of data. """
    pylab.hold(True)

    pylab.plot([p[0] for p in class_a], [p[1] for p in class_a], 'bo')
    pylab.plot([p[0] for p in class_b], [p[1] for p in class_b], 'ro')


def create_p_matrix(data, kernel_function):
    """ Creates an N x N matrix P.
        P_ij = t_i * t_j * K(x_i, x_j)
        N is the number of data points.
        K is a kernel function.
        t is the class (-1 or 1).
        x is a vector with data points.
    """

    N = len(data)
    P = numpy.zeros(shape=(N, N))

    for i in range(N):
        for j in range(N):
            t_i = data[i][2]
            t_j = data[j][2]
            x_i = data[i][:2]
            x_j = data[j][:2]

            P[i, j] = t_i * t_j * kernel_function(x_i, x_j)

    return P


def create_q_and_h_vectors(N):
    """ Creates the q and h vectors necessary for
        calling the qp function and finding an optimal
        alpha, as stated in the beginning of this file.
    """

    q = numpy.empty(N)
    q.fill(-1)

    h = numpy.zeros(N)

    return q, h


def create_g_matrix(N):
    """ Creates the G matrix necessary for
    calling the qp function and finding an optimal
    alpha, as stated in the beginning of this file.
    """

    G = numpy.zeros(shape=(N, N))
    numpy.fill_diagonal(G, -1)

    return G


def find_optimal_alphas(data, kernel_function):
    """ Calls the qp function and finds an optimal
        alpha, as explained in the beginning of
        this file.
    """

    N = len(data)

    q, h = create_q_and_h_vectors(N)
    G = create_g_matrix(N)

    P = create_p_matrix(data, kernel_function)

    # Call qp. This returns a dictionary data structure. The index 'x' contains the alpha values.
    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    alphas = list(r['x'])

    print(alphas)

    return alphas


def pick_non_zero_alphas_and_create_indicator_list(data, alphas):
    indicator_list = []
    threshold = 10e-5

    for i in range(len(alphas)):
        alpha = alphas[i]

        if alpha > threshold:
            x = data[i][0]
            y = data[i][1]
            t = data[i][2]

            values = (x, y, t, alpha)
            indicator_list.append(values)

    print("Found " + str(len(alphas)) + " alphas in total.")
    print("Found " + str(len(indicator_list)) + " that were non-zero.")

    return indicator_list


def indicator_function(x_star, y_star, indicator_list, kernel_function):
    """ The indicator function can classify new data
        points x* = (x, y). If positive, the class is 1. If
        negative, the class is -1. A value
        between -1 and 1 lies on the margin and this
        should not happen. The t_i is the class and
        x_i is the data point vector.

        ind(x*) = sum( alpha_i * t_i * K(x*, x_i) )
    """

    N = len(indicator_list)
    sum = 0

    for i in range(N):
        # The indicator_list contains the alpha, the class (t) and data points (x, y).
        alpha_i = indicator_list[i][3]
        t_i = indicator_list[i][2]
        x_i = indicator_list[i][:2]

        sum += alpha_i * t_i * kernel_function([x_star, y_star], x_i)

    return sum


def plot_decision_boundary(indicator_list, kernel_function):
    """ Plots the decision boundary of the classification. """

    x_range = numpy.arange(-4, 4, 0.05)
    y_range = numpy.arange(-4, 4, 0.05)

    grid = matrix([[indicator_function(x, y, indicator_list, kernel_function) for y in y_range] for x in x_range])

    pylab.contour(x_range, y_range, grid,
                  (-1.0, 0.0, 1.0),
                  colors=('red', 'black', 'blue'),
                  linewidths=(1, 3, 1))


def plot_points_on_margins(indicator_list):
    """ Plots the points on the margin.
        This means that they are the support
        vectors and that their alphas
        are non-zero.
    """
    pylab.plot([p[0] for p in indicator_list], [p[1] for p in indicator_list], 'go')


def run():
    # The chosen kernel function.
    kernel_function = radial_basis_kernel

    # Create random binary classified test data.
    class_a, class_b = create_random_classified_test_data()

    # Merge the data into one dataset and shuffle it.
    data = class_a + class_b
    random.shuffle(data)

    # Find the optimal alphas with the given kernel function and data.
    alphas = find_optimal_alphas(data, kernel_function)
    indicator_list = pick_non_zero_alphas_and_create_indicator_list(data, alphas)

    # Plot the data points and the found decision boundary.
    plot_data_points(class_a, class_b, indicator_list, kernel_function)
    plot_decision_boundary(indicator_list, kernel_function)
    plot_points_on_margins(indicator_list)
    pylab.show()


run()
