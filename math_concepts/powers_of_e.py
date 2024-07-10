import numpy as np

def power_of_e(x):
    '''
    This method calculates the approximate value of e^x by using the following identity:
    e^x = 1+x/1!+x^2/2!+...
    :param x: An integer to whose power e should be raised
    :return: Approxmate value of e^x
    '''

    # The identity above will be evaluated for 1000 terms
    factorial = 1
    result = 0
    for i in range(1000):
        if i > 0:
            factorial = factorial * i
        result += pow(x,i)/factorial

    return result

# Test code

assert(round(power_of_e(1),10) == round(np.e, 10))
assert(round(power_of_e(2),10) == round(pow(np.e, 2), 10))
assert(round(power_of_e(-2),10) == round(pow(np.e, -2), 10))
assert(round(power_of_e(10),10) == round(pow(np.e, 10), 10))
