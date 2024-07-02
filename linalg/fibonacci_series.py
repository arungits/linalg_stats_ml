import numpy as np
import numpy.linalg as LA

def get_nth_fibonacci_term(n):
    # This method returns the nth term in the Fibonacci series 0, 1, 1, 2, 3, 5, ...
    # by using powers of A computed using eigen decomposition
    # Let u0 = [1, 0] which is a vector with 2nd term and 1st term of the Fibonacci series
    # To compute u1 (vector with 3rd and 2nd terms of the series, we can find the dot product of
    # A.u0 where A = [[1, 1], [1, 0]]
    # To compute u(n) (vector with n+2 and n+1 terms of the series, we can find the dot product of
    # A**n.u0
    # If we can express u0 in terms of linear combination of the eigenvectors of A or in other words,
    # find c = (c1, c2) such that Ec = u0, where E is the eigenvectors matrix of A,
    # then we can easily compute u(n) = A**n.u0 using the formula,
    # c1*(eig1**n)*(eigvector1) + c2*(eig2**n)*(eigvector2) where eig1 and eig2 are eigen values of A

    # Check n >= 2
    assert(n>=2)

    # Initialize u0 and A
    u0 = np.array([1, 0])
    A = np.array([[1,1], [1,0]])

    # Diagonalize matrix A using Eigen decomposition
    eigenvalues, E = LA.eig(A)

    # Find solution for Ec = u0
    c = LA.solve(E, u0)


    # Find nth term of the Fibonacci series, find u(n-2)
    # Step1: find the linear combination of u(n-2) with respect to E
    l = np.array([c[0] * pow(eigenvalues[0], n-2), c[1] * pow(eigenvalues[1], n-2)])
    # Step2: take the dot product of E and the above vector to find the nth term of the Fib series
    nth_term = E@l
    return np.round(nth_term[0],2)

# Test code

assert(get_nth_fibonacci_term(2) == 1)
assert(get_nth_fibonacci_term(3) == 1)
assert(get_nth_fibonacci_term(4) == 2)
assert(get_nth_fibonacci_term(5) == 3)
assert(get_nth_fibonacci_term(6) == 5)
assert(get_nth_fibonacci_term(10) == 34)
assert(get_nth_fibonacci_term(20) == 4181)
assert(get_nth_fibonacci_term(30) == 514229)
assert (get_nth_fibonacci_term(40) == 63245986)
assert(get_nth_fibonacci_term(50) == 7778742049)
assert(get_nth_fibonacci_term(60) == 956722026041)
# Beyond 60, due to numerical accuracy issues in computing eigvalues and eigvectors, this method
# does not generate accurate results