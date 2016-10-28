import numpy as np
from numpy.linalg import eigvals

# entanglement entropy
def entanglement_entropy(reduceddensitymatrix):
    eigenvalues = eigvals(reduceddensitymatrix)
    eigenvalues = np.real(eigenvalues)
    entropy = np.sum(- eigenvalues*np.log(eigenvalues))
    return entropy

# participation ratio
def participation_ratio(reduceddensitymatrix):
    eigenvalues = eigvals(reduceddensitymatrix)
    eigenvalues = np.real(eigenvalues)
    K = 1./np.sum(eigenvalues*eigenvalues)
    return K

# negativity
def negativity(reduceddensitymatrix):
    sqmat = np.dot(np.transpose(np.conj(reduceddensitymatrix)), reduceddensitymatrix)
    sqmat_eigvals = eigvals(sqmat)
    negval = np.sum(np.sqrt(np.real(sqmat_eigvals)))
    return 0.5*(negval-1)