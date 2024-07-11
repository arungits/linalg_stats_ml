import numpy as np

def loop_condition(sign, x, current_x):
    if sign < 0:
        return current_x > x
    else:
        return current_x < x

def solve_de_by_eulers_method(dy_dx, d2y_dx2, x, initial_condition):
    '''
    This method implements Euler's method to solve for y in differential equations when
    y' and y'' are given. This method uses quadratic approximation instead of linear approximation
    for getting more accurate results.
    :param dy_dx: The first derivative of y expressed as a function of (x,y)
    :param d2y_dx2: The second derivative of y expressed as a function of (x,y, dy_dx)
    :param x: The value of x for which y must be found using Euler's method
    :param initial_condition: A tuple specifying the initial condition (x0, y0)
    :return: Returns the value of y corresponding to x after solving the DE using Euler's method
    '''
    current_x, current_y = initial_condition
    sign = -1 if x < current_x else 1
    step = 0.01 * sign
    while loop_condition(sign, x, current_x):
        first_derivative = dy_dx(current_x, current_y)
        second_derivative = d2y_dx2(current_x, current_y)
        new_x = current_x + step
        new_y = current_y + first_derivative * step + second_derivative * pow(step, 2) / 2
        current_x, current_y = new_x, new_y
    return current_y

# Test code

def execute_test(dy_dx, d2y_dx2, x, initial_condition, actual_solution):
    estimate = solve_de_by_eulers_method(dy_dx, d2y_dx2, x, initial_condition)
    actual = actual_solution(5)
    error = (actual - estimate) / actual
    print(f"Estimate: {estimate}, Actual: {actual}, Error: {round((error * 100),2)}%")
    assert(abs(error) < 0.05)

# Test for the DE: dy/dx = x + y for the initial condition (0,1)
dy_dx = lambda x, y: x+y
d2y_dx2 = lambda x, y: 1 + x + y
initial_condition = (0,1)
actual_solution = lambda x: 2 * pow(np.e, x) - x - 1 # For the given initial condition (0,1)
x_to_solve_for = 5
execute_test(dy_dx, d2y_dx2, x_to_solve_for, initial_condition, actual_solution)

# Test for the DE: dy/dx = xy for the initial condition (0,1)
dy_dx = lambda x, y: x*y
d2y_dx2 = lambda x, y: pow(x, 2) * y + y
initial_condition = (0,1)
actual_solution = lambda x: pow(np.e, pow(x, 2) / 2) # For the given initial condition (0,1)
x_to_solve_for = 5
execute_test(dy_dx, d2y_dx2, x_to_solve_for, initial_condition, actual_solution)
x_to_solve_for = -5
execute_test(dy_dx, d2y_dx2, x_to_solve_for, initial_condition, actual_solution)



