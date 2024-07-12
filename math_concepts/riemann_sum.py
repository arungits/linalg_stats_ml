import numpy as np

def riemann_sum(f, low, high):
    '''
    This method implement Riemann sum to compute the area under the curve y=f(x) between the limits
    x=low and x=high. This is same as the definite integral of y=f(x) in the interval (low, high)
    according to the First Fundamanetal Theorem of Calculus.
    :param f: Function y=f(x)
    :param low: Lower limit from which the area under f(x) needs to be computed
    :param high: Upper limit to which the area under f(x) needs to be computed
    :return: Riemann sum of the area under the curve f(x) between x=low and x=high
    '''
    # Compute the number of pieces to divide the area between low and high
    # Higher n the more accurate the area computed will be
    n = int(max(1000, abs(high - low)/0.5))
    width = (high - low) / n
    total_area = 0
    x = low
    for i in range(n):
        new_x = x + width
        mid_x = (x + new_x) / 2
        f_at_mid = f(mid_x)
        area = f_at_mid * width
        total_area += area
        x = new_x
    return total_area

# Test code

def execute_test(f, actual_area, low, high):
    estimate = riemann_sum(f, low, high)
    actual = actual_area(low, high)
    error = None
    if actual != 0:
        error = abs(estimate - actual) / actual
        # Error in percentage
        error = round(error * 100, 2)
    print(f"Estimate: {estimate}, Actual: {actual}, Error: {error}%")
    assert(round(actual) == round(estimate) or error < 0.05)

f = lambda x: x
actual_area = lambda low, high: (pow(high, 2) - pow(low, 2)) / 2
execute_test(f, actual_area, 0, 1)
execute_test(f, actual_area, 1, 0)
execute_test(f, actual_area, -1, 1)

f = lambda x: 5
actual_area = lambda low, high: 5 * (high - low)
execute_test(f, actual_area, -10, 10)

f = lambda x: 3*pow(x,2)
actual_area = lambda low, high: pow(high, 3) - pow(low, 3)
execute_test(f, actual_area, 0, 10)
execute_test(f, actual_area, -10, 10)

# Natural Logarithm of x is defined as the definite integration of f(x)=1/x between limits 1 and x
# f is the definition of logarithm of x, where x > 0
f = lambda x: 1/x # defined for x > 0
actual_area = lambda low, high: np.log(high)
execute_test(f, actual_area, 1, 10)
execute_test(f, actual_area, 1, 100)
execute_test(f, actual_area, 1, 1)
execute_test(f, actual_area, 1, 0.5)
execute_test(f, actual_area, 1, 0.9)
execute_test(f, actual_area, 1, 0.99)

# Bell curve f = e^(-x^2) has an area of sqrt(pi) between -infinity and infinity
f = lambda x: pow(np.e, -1*pow(x,2))
actual_area = lambda x,y: np.sqrt(np.pi) # Parameters are not used as the area is known
execute_test(f, actual_area, -10000, 10000) #Limits are set to a large number as we cannot compute till inifinity
