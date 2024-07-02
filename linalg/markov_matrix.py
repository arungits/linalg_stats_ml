import numpy as np

def markov_matrix(i, M, trials):
    # i -Vector representing the initial condition
    # M - Markov matrix with probabilities that add up to 1 to predict the changed vecotr v from i at the end of a trial
    # trials - No of trials

    m, n = M.shape
    # Make sure M is a square matrix so that we can perform Eigen decompoosition of M
    assert(m == n)
    assert(len(i) == m)

    # Eigen decomposition of M
    evals, S = np.linalg.eig(M)
    # Make sure that M is a markov matrix by checking if 1 is an eigenvalue of M
    assert(1 in np.round(evals))

    # Compute the change in state after "trials" trials by computing v=(A**trials)@i
    # (A**trials) can be easily computed as A**trials = S@(evals**trials)@inv(S)
    # where S is the eigenvectors matrix
    S_inv = np.linalg.inv(S)
    D = np.diag(np.pow(evals, trials))
    An = S@D@S_inv

    result_after_trials = An@i
    return result_after_trials, np.round(evals, 2), S

# Test code

#Initial condition
initial = np.array([0,1000])
# Markov matrix
M = np.array([[0.9, 0.2], [0.1, 0.8]])

result, evals, evecs = markov_matrix(initial, M, 100)
index = np.where(evals == 1)[0][0]
# Get the eigen vector corresponding to the eigenvalue of 1
v = evecs[:,index]

# Make sure that the steady state for the system represented by the Markov Matrix M is given by
# the eigenvector corresponding to the eigenvalue of 1 after a large number of trials (say 100 trials)

assert(np.round(np.sum(initial) * v[0]/np.sum(v),2) == np.round(result[0], 2))
assert(np.round(np.sum(initial) * v[1]/np.sum(v),2) == np.round(result[1], 2))
