from itertools import product

import numpy as np
from numpy.linalg import eig

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

# Schmidt decomposition
def schmidt_decomposition(bipartitepurestate_tensor):
    state_dims = bipartitepurestate_tensor.shape
    mindim = np.min(state_dims)

    rho1 = bipartitepurestate_reduceddensitymatrix(bipartitepurestate_tensor, 1)
    eigenvalues1, unitarymat1 = eig(rho1)
    inv_unitarymat1 = np.linalg.inv(unitarymat1)
    coefmat0 = np.dot(bipartitepurestate_tensor, inv_unitarymat1)

    decomposition = [(float(np.real(eigenvalues1[k])),
                      coefmat0[:, k]/np.sqrt(np.real(eigenvalues1[k])),
                      unitarymat1[:, k])
                     for k in range(mindim)]
    decomposition = sorted(decomposition, key=lambda dec: dec[0], reverse=True)

    return decomposition