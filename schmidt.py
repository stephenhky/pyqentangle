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

