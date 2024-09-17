import numpy as np
import numpy.linalg as LA

def compute_gradient(gradient_f, current):
    gradient = gradient_f(*current)
    return gradient

def is_a_minima(hessian_matrix, current):
    # This method performs the second derivative test
    hessian = hessian_matrix(*current)
    evals, _ = LA.eig(hessian)
    # If the hessian matrix at the current point is positive definite (i.e. all eigen values are +ve)
    # Then the point is a local minima
    return np.all(evals > 0)
def find_minima(gradient_f, hessian_matrix, initial_solution, max_iterations = 100000):
    '''
    :param gradient_f: Gradient of function f
    :param hessian_matrix: Hessian matrix of f that contains functions to compute second derivatives
    :param initial_solution: Initial solution to use for seeding the algorithm as tuple
    :return: Return a local minima of f or None if local minima couldn't be found
    '''
    current = initial_solution
    prev_gradient, gradient = None, None
    step = 0.001
    for i in range(max_iterations):
        prev_gradient = gradient
        gradient = compute_gradient(gradient_f, current)
        if np.all(np.abs(gradient) <= 0.001):
            # The current solution is a minima or maxima or saddle point if gradient is zero
            if is_a_minima(hessian_matrix, current):
                # If the current point is a minima, return the point
                return current
            current -= step * prev_gradient
        else:
            current = current - step * gradient
    # Local minima for f couldn't be found
    return None

# Test code
f = lambda x,y: pow(x,4) + pow(y,4)
gradient_f = lambda x, y: np.array([4*pow(x,3), 4*pow(y,3)])
hessian_matrix = lambda x, y: np.array([[12*pow(x,2), 0], [0, 12*pow(y,2)]])
initial_solution = (10,10)
print(find_minima(gradient_f, hessian_matrix, initial_solution))