import numpy as np

def orthonormalize_basis(A):
    # NOTE: This uses Gram Schmidt process for normalizing basis
    # NOTE: THIS METHOD IS NOT NUMERICALLY STABLE AND IS FOR EDUCATIONAL PURPOSES ONLY

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
A = np.random.randn(3,3)
# Orthonormal Matrix
Q = orthonormalize_basis(A)
for i in range(len(Q)):
    for j in range(i+1, len(Q)):
        assert(np.round(Q[i]@Q[j], 2) == 0)

