from itertools import product

import numpy as np
from numpy.linalg import eig

def bipartitepurestate_densitymatrix(bipartitepurestate_tensor):
    state_dims = bipartitepurestate_tensor.shape
    rho = np.zeros(state_dims*2, dtype=np.complex)
    for i, j, ip, jp in product(*map(range, state_dims*2)):
        rho[i, j, ip, jp] = bipartitepurestate_tensor[i, j]*np.conj(bipartitepurestate_tensor[ip, jp])
    return rho

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

def entanglement_entropy(bipartitepurestate_tensor):
    state_dims = bipartitepurestate_tensor.shape
    which_mindim = np.argmin(state_dims)
    rho = bipartitepurestate_reduceddensitymatrix(bipartitepurestate_tensor, which_mindim)
    eigenvalues, _ = eig(rho)
    eigenvalues = np.real(eigenvalues)
    entropy = np.sum(- eigenvalues*np.log(eigenvalues))
    return entropy

def schmidt_decomposition(bipartitepurestate_tensor):
    state_dims = bipartitepurestate_tensor.shape
    mindim = np.min(state_dims)

    rho0 = bipartitepurestate_reduceddensitymatrix(bipartitepurestate_tensor, 0)
    rho1 = bipartitepurestate_reduceddensitymatrix(bipartitepurestate_tensor, 1)

    eigenvalues0, unitarymat0 = eig(rho0)
    eigenorder0 = np.argsort(eigenvalues0)
    eigenvalues1, unitarymat1 = eig(rho1)
    eigenorder1 = np.argsort(eigenvalues1)

    decomposition = [(eigenvalues0[eigenorder0[orderid]],
                      unitarymat0[:, eigenorder0[orderid]],
                      unitarymat1[:, eigenorder1[orderid]]) for orderid in range(mindim)]

    return decomposition