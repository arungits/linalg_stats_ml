import numpy as np
import numpy.linalg as LA

def check_orthonormality(Q):
    for i in range(len(Q)):
        for j in range(i + 1, len(Q)):
            dot_product = np.round(Q[i] @ Q[j], 2)
            # Check if the dot_product is aribitrarily close to 0
            # Dot product may not be exactly zero as this algorithm is not numerically stable
            assert (abs(dot_product - 0) < 0.02)


def orthonormalize_basis(A):
    # NOTE: This uses Gram Schmidt process for normalizing basis
    # NOTE: THIS METHOD IS NOT NUMERICALLY STABLE AND IS FOR EDUCATIONAL PURPOSES ONLY

    # Check it's a square matrix with full rank
    m, n = A.shape
    assert(m == n)
    assert(LA.matrix_rank(A) == m)

    m, n = A.shape
    orthonormal_basis = []
    for col in range(n):
        v = A[:, col] # Column vector
        o_v = v # Orthonormalized vector
        for basis in orthonormal_basis:
            o_v = o_v - (v@basis) * basis
        orthonormal_basis.append(o_v / np.sqrt(v@v))
    return orthonormal_basis

# Test code
A = np.array([[3.0, 1.0], [2.0, 2.0]])
B = np.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]])

# Orthonormal Matrix
check_orthonormality(orthonormalize_basis(A))
check_orthonormality(orthonormalize_basis(B))

