import numpy as np
import numpy.linalg as LA

def orthonormalize_basis(A):
    # NOTE: This uses Gram Schmidt process for normalizing basis
    # Check it's a square matrix with full rank before proceeding
    m, n = A.shape
    assert(m == n)
    assert(LA.matrix_rank(A) == m)

    m, n = A.shape
    orthonormal_basis = []
    for col in range(n):
        v = A[:, col] # Column vector
        o_v = v # Orthonormal form of vector v
        for basis in orthonormal_basis:
            o_v = o_v - (v@basis) * basis
        orthonormal_basis.append(o_v / np.sqrt(o_v@o_v))
    return orthonormal_basis


# Test code

def check_orthonormality(Q):
    # Check that all columns are unit vectors
    for i in range(len(Q)):
        assert(np.round(Q[i]@Q[i], 2) == 1)
    # Check that all columns are orthogonal to all other columns
    for i in range(len(Q)):
        for j in range(i + 1, len(Q)):
            dot_product = np.round(Q[i] @ Q[j], 2)
            assert(dot_product == 0)
    # Check that the inverse of the matrix is same as its transpose
    Q = np.array(Q)
    assert(np.all(np.round(Q.transpose(), 2) == np.round(LA.inv(Q), 2)))

def test_orthogonalization():
    A = np.array([[3.0, 1.0], [2.0, 2.0]])
    B = np.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]])
    # Check the result of Gram Schmidt process is an orthogonal matrix
    check_orthonormality(orthonormalize_basis(A))
    check_orthonormality(orthonormalize_basis(B))

    for i in range(10):
        A = np.random.randn(10, 10)
        # Make sure A is a non singular matrix with full rank before apply Gram Schmidt
        while (LA.matrix_rank(A) != 10):
            A = np.random.randn(10, 10)
        # Check the result of Gram Schmidt process is an orthogonal matrix
        check_orthonormality(orthonormalize_basis(A))

test_orthogonalization()

