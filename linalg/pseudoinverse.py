import numpy as np
from linalg.svd import svd

def pinv(A):
    m,n = A.shape
    U,S,Vt = svd(A)
    # Step1: Find the pseudoinverse of S
    S_pinv = S.transpose()
    for i in range(len(S_pinv)):
        if i >= m:
            break
        if np.round(S_pinv[i][i]) == 0:
            continue
        S_pinv[i][i] = 1/S_pinv[i][i]
    # By SVD, we know A = U@S@Vt
    # Therefore A_pinv = V@S_pinv@U_transpose
    A_pinv = Vt.transpose()@S_pinv@U.transpose()
    return A_pinv

# Test code

for i in range(100):
    A = np.random.randn(10,2)
    A_pinv = pinv(A)
    # Check the shape of A's pseudoinverse
    assert(A_pinv.shape == (2,10))
    # Check that A_pinv@A@A_pinv is A_pinv
    R1 = A_pinv@A@A_pinv
    assert(np.all(np.round(R1,2) == np.round(A_pinv, 2)))
    # Check that A@A_pinv@A is A
    R2 = A@A_pinv@A
    assert(np.all(np.round(R2, 2) == np.round(A, 2)))

for i in range(100):
    A = np.random.randn(2,10)
    A_pinv = pinv(A)
    # Check the shape of A's pseudoinverse
    assert(A_pinv.shape == (10,2))
    # Check that A_pinv@A@A_pinv is A_pinv
    R1 = A_pinv@A@A_pinv
    assert(np.all(np.round(R1,2) == np.round(A_pinv, 2)))
    # Check that A@A_pinv@A is A
    R2 = A@A_pinv@A
    assert(np.all(np.round(R2, 2) == np.round(A, 2)))