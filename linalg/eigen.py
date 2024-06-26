import numpy as np

A = np.random.randn(40,40)
u,d,w = np.linalg.svd(A)


I = np.identity(2)
print(I)

S = np.random.randn(2,2)

print(S)

print(I@S)