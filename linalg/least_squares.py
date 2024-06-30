import numpy as np
import numpy.linalg as LA

def least_squares_fit_by_projection(A, b):
    # Use the following matrix to project b onto the column space of A
    # p = A@inv(A_transpose@A)@A_transpose@b
    # Then solve for x using the equation Ax = p using pseudoinverse of A
    # where x is the best fit solution for Ax = b
    A_transpose = A.transpose()
    p = A @ LA.inv(A_transpose @ A) @ A_transpose @ b
    x = LA.pinv(A) @ p
    return x

def least_squares_fit_by_direct_formula(A, b):
    # This method is similar to least_squares_fit_by_projection
    # Here instead of solving Ax = p by elimination, we directly compute x as
    # dot product of inv(A_transpose@A)@A_transpose@b
    A_transpose = A.transpose()
    x = LA.inv(A_transpose @ A) @ A_transpose @ b
    return x


# Test code
# Repeat for 10 trials
for i in range(10):
    # Generate a random 10x2 matrix A and a 10 dimensional vector b
    A = np.random.randn(10,2)
    b = np.random.randn(10)
    # Find the best fit Least Squares solution using pseudoinverse as well as computing directly
    # Compare the results to make sure both methods return the same x.
    x_projection = np.round(least_squares_fit_by_projection(A, b),4)
    x_direct = np.round(least_squares_fit_by_direct_formula(A, b), 4)
    # Ensure both methods return the same x
    assert(np.all(x_projection == x_direct))



