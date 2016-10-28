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

# fixing the phase
def phase_fixing(bipartitepurestate_tensor, decomposed):
    numcomps = len(decomposed)
    state_dims = bipartitepurestate_tensor.shape
    idxcombs = list(product(*map(range, state_dims)))
    idxcombs = filter(lambda idxcomb: bipartitepurestate_tensor[idxcomb]!=0 and bipartitepurestate_tensor[idxcomb]!=1.+0.j,
                      idxcombs)
    print idxcombs
    picked_idxcombs = [idxcombs[idx] for idx in np.random.randint(len(idxcombs), size=numcomps)]

    A = np.zeros((numcomps, numcomps))
    b = np.zeros((numcomps, 1))

    for kp, (i, j) in zip(range(numcomps), picked_idxcombs):
        b[kp, 0] = bipartitepurestate_tensor[i, j]
        for k in range(numcomps):
            coef, vecA, vecB = decomposed[k]
            A[kp, k] = np.sqrt(coef)*vecA[i]*vecB[j]

    print A
    print b
    cmpfctors = solve(A, b)

    decomposition = [(np.sqrt(coef)*cmpfactor, vecA, vecB)
                     for cmpfactor, (coef, vecA, vecB) in zip(cmpfctors, decomposed)]

    return decomposition

