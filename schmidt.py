from itertools import product

import numpy as np
from numpy.linalg import eig
from scipy.linalg import solve

# total density matrix
def bipartitepurestate_densitymatrix(bipartitepurestate_tensor):
    state_dims = bipartitepurestate_tensor.shape
    rho = np.zeros(state_dims*2, dtype=np.complex)
    for i, j, ip, jp in product(*map(range, state_dims*2)):
        rho[i, j, ip, jp] = bipartitepurestate_tensor[i, j]*np.conj(bipartitepurestate_tensor[ip, jp])
    return rho

# reduced density matrix. kept=0 for the first system, 1 the second.
def bipartitepurestate_reduceddensitymatrix(bipartitepurestate_tensor, kept):
    state_dims = bipartitepurestate_tensor.shape
    if not (kept in [0, 1]):
        raise ValueError('kept can only be 0 or 1!')
    rho = np.zeros((state_dims[kept],)*2, dtype=np.complex)
    for i, ip in product(*map(range, (state_dims[kept],)*2)):
        if kept == 0:
            rho[i, ip] = np.sum([bipartitepurestate_tensor[i, j]*np.conj(bipartitepurestate_tensor[ip, j])
                                 for j in range(state_dims[1])])
        else:
            rho[i, ip] = np.sum([bipartitepurestate_tensor[j, i]*np.conj(bipartitepurestate_tensor[j, ip])
                                 for j in range(state_dims[0])])
    return rho

# entanglement entropy
def entanglement_entropy(bipartitepurestate_tensor):
    state_dims = bipartitepurestate_tensor.shape
    which_mindim = np.argmin(state_dims)
    rho = bipartitepurestate_reduceddensitymatrix(bipartitepurestate_tensor, which_mindim)
    eigenvalues, _ = eig(rho)
    eigenvalues = np.real(eigenvalues)
    entropy = np.sum(- eigenvalues*np.log(eigenvalues))
    return entropy

# Schmidt decomposition
def schmidt_decomposition(bipartitepurestate_tensor):
    state_dims = bipartitepurestate_tensor.shape
    mindim = np.min(state_dims)

    rho0 = bipartitepurestate_reduceddensitymatrix(bipartitepurestate_tensor, 0)
    rho1 = bipartitepurestate_reduceddensitymatrix(bipartitepurestate_tensor, 1)

    eigenvalues0, unitarymat0 = eig(rho0)
    eigenorder0 = np.argsort(eigenvalues0)
    eigenvalues1, unitarymat1 = eig(rho1)
    eigenorder1 = np.argsort(eigenvalues1)

    decomposition = [(float(np.real(eigenvalues0[eigenorder0[orderid]])),
                      unitarymat0[:, eigenorder0[orderid]],
                      unitarymat1[:, eigenorder1[orderid]]) for orderid in range(mindim-1, -1, -1)]

    return decomposition

def schmidt_decomposition_correctphase(bipartitepurestate_tensor):
    state_dims = bipartitepurestate_tensor.shape
    mindim = np.min(state_dims)

    rho1 = bipartitepurestate_reduceddensitymatrix(bipartitepurestate_tensor, 1)
    eigenvalues1, unitarymat1 = eig(rho1)
    inv_unitarymat1 = np.linalg.inv(unitarymat1)

    decomposition = []
    for k in range(mindim):
        vec0 = np.zeros(state_dims[0])
        for i in range(state_dims[0]):
            vec0[i] = np.sum([bipartitepurestate_tensor[i, j]*inv_unitarymat1[j, k] for j in range(state_dims[1])])
        decomposition += [(float(np.real(eigenvalues1[k])),
                           vec0,
                           unitarymat1[:, k])]

    return decomposition