import numpy as np
import numpy.linalg as LA

def sorted_eigen(eigendecomposition):
    eigenvalues = eigendecomposition[0]
    eigenvectors = eigendecomposition[1]
    ids = np.argsort(eigenvalues)[::-1]
    return (eigenvalues[ids], eigenvectors[:,ids])

def svd(A):
    # This method implements SVD from scratch for a mxn matrix
    # To perform singular value decomposition for the matrix A and find the components, viz.,
    #  U (Column space), S (diagonal matrix) and V (Rowspace), we will do the following
    # 1. Eigen decomposition / diagonalization of the positive definite n x n matrix
    # A_transpose @ A to find its eigenvalues and eigenvectors to get V & S
    # 2. Solve A = U@sqrt(S)@V_transpose to find U

    m, n = A.shape
    A_transpose = A.transpose()

    # Eigen decomposition of A_transpose@A
    E, V = sorted_eigen(LA.eig(A_transpose @ A))
    # Singular values of A are square roots of the eigenvalues of A_tranpose@A
    E = np.sqrt(np.round(E, 4))
    S = np.zeros((m, n))
    for i in range(len(E)):
        if np.round(E[i]) == 0:
            continue
        S[i,i] = E[i]
    # Now solve for U by solving U = A@V@S_inv (As per SVD, A = U@S@V_transpose
    S_inv = np.zeros((n, m))
    for i in range(len(E)):
        if np.round(E[i]) == 0:
            continue
        S_inv[i,i] = 1/E[i]
    U = A@V@S_inv
    return (U, S, V.transpose())

# Test code
for i in range(100):
    A = np.random.randn(10,2)
    U,S,Vt = svd(A)
    A_reconstructed = U@S@Vt
    assert(np.all(np.round(A, 2) == np.round(A_reconstructed, 2)))

for i in range(100):
    A = np.random.randn(2, 10)
    U,S,Vt = svd(A)
    A_reconstructed = U@S@Vt
    assert(np.all(np.round(A, 2) == np.round(A_reconstructed, 2)))





