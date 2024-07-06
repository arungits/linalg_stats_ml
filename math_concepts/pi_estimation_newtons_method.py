import numpy as np

def find_pi():
    # This method finds pi using Newton's method
    # We know that pi is one of the roots of the function sin(x)
    # We will use Newton's method to find the roots of sin(x) by starting with an initial guess of 3.14
    # Newton's method is an iterative process using the formula:
    # x_n+1 = x_n - f(x_n) / f'(x_n)
    # The above process is repeated until x's converge

    # Function: sin(x)
    f_x = lambda x: np.sin(x)
    # Derivative of f_x
    f_prime_x = lambda x: np.cos(x)

    guess = 3.14
    epsilon = pow(10, -10) # To test for convergence
    # Repeat for 100 times or until convergence whichever is smaller
    for i in range(100):
        new_guess = guess - f_x(guess)/f_prime_x(guess)
        if abs(guess - new_guess) <= epsilon:
            print(f"Found answer in {i} iterations")
            return guess
        guess = new_guess

    raise "Unable to find the value of pi"

assert(round(find_pi(), 10) == 3.1415926536)