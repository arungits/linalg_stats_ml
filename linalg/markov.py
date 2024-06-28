import numpy as np

initial = np.array([0,1000])

A = np.array([[0.9, 0.2], [0.1, 0.8]]) # Markov probability matrix

evals, evecs = np.linalg.eig(A)

print(evecs)
S = evecs
S_inv = np.linalg.inv(S)

#After 100 iterations

evals_new = np.array([[pow(evals[0], 100),0],[0, pow(evals[1],100)]])
A_after_100 = S.dot(evals_new).dot(S_inv).dot(initial)
print(A_after_100)

v = np.array([[1,2,3]])
print(v.dot(v.transpose()))
