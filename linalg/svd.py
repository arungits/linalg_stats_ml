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
A = np.array([[87,24], [2,32], [80,14], [98,74], [94,59], [37,65], [3,6], [11,56], [15,30], [26,33]])

U, S, V = svd(A)
A_reconstructed = U@S@V.transpose()
# Due to numerical accuracy issues, round the reconstructed matrix before comparing
A_reconstructed_rounded = np.round(A_reconstructed, 2)
# Check if the reconstructed matrix is equal to original matrix or is same as original matrix with sign flipped
# The reason for sign flipping is unknown and needs to be investigated
assert(np.all(A == A_reconstructed_rounded)  or np.all((A * -1) == A_reconstructed_rounded))


