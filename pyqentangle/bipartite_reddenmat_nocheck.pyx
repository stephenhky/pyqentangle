
import numpy as np
cimport numpy as np

def bipartitepurestate_reduceddensitymatrix_nocheck(np.ndarray bipartitepurestate_tensor, int kept):
    cdef int state_dims_0 = bipartitepurestate_tensor.shape[0]
    cdef int state_dims_1 = bipartitepurestate_tensor.shape[1]

    cdef int kept_dim, integrate_dim
    if kept == 0:
        kept_dim, integrate_dim = state_dims_0, state_dims_1
    else:
        kept_dim, integrate_dim = state_dims_1, state_dims_0

    cdef np.ndarray rho = np.zeros((kept_dim,) * 2, dtype=np.complex)
    cdef int i, ip, j

    for i in range(kept_dim):
        for ip in range(kept_dim):
            rho[i, ip] = 0
            for j in range(integrate_dim):
                if kept == 0:
                    rho[i, ip] += bipartitepurestate_tensor[i, j] * np.conj(bipartitepurestate_tensor[ip, j])
                else:
                    rho[i, ip] += bipartitepurestate_tensor[j, i] * np.conj(bipartitepurestate_tensor[j, ip])

    return rho
