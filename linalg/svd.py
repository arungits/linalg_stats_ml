import numpy as np
import numpy.linalg as LA

def sorted_eigen(eigendecomposition):
    eigenvalues = eigendecomposition[0]
    eigenvectors = eigendecomposition[1]
    ids = np.argsort(eigenvalues)[::-1]
    return (eigenvalues[ids], eigenvectors[:,ids])
def svd(A):
    # NOTE: THIS METHOD IS NOT NUMERICALLY STABLE AND IS FOR EDUCATIONAL PURPOSES ONLY
    # This method implements SVD from scratch for rectangular matrix m x n where m > n
    # To perform singular value decomposition for the matrix A and find the components, viz.,
    #  U (Column space), S (diagonal matrix) and V (Rowspace), we will do the following
    # 1. Eigen decomposition / diagonalization of the positive definite n x n matrix
    # A_transpose @ A to find its eigenvalues and eigenvectors to get V & S
    # 2. Eigen decomposition of the positive definite m x m matrix
    # A @ A_transpose to find its eigenvectors which is U
    m, n = A.shape
    # Asset the matrix is retangular
    assert (m > n)
    A_transpose = A.transpose()
    _, V = sorted_eigen(LA.eig(A_transpose @ A))
    eigenvalues, U = sorted_eigen(LA.eig(A @ A_transpose))
    S = np.diag(np.sqrt(np.round(eigenvalues, 4)))
    S = np.delete(S, range(n, m), 1)
    return (U, S, V)

# Test code
A = np.random.randn(10,2)
U, S, V = svd(A)
A_reconstructed = U@S@V.transpose()
A_abs_rounded = np.abs(np.round(A, 2))
A_reconstructed_abs_rounded = np.abs(np.round(A_reconstructed, 2))
assert(np.all(A_abs_rounded == A_reconstructed_abs_rounded))

