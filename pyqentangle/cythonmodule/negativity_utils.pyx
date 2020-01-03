

import numpy as np
cimport numpy as np


def bipartitepurestate_partialtranspose_densitymatrix_cython(np.ndarray bipartitepurestate_tensor, int pt_subsys):
    cdef int i, j, ip, jp
    cdef int dim0 = bipartitepurestate_tensor.shape[0]
    cdef int dim1 = bipartitepurestate_tensor.shape[1]

    cdef np.ndarray rho = np.zeros((dim0, dim1, dim0, dim1), dtype=np.complex)

    for i in range(dim0):
        for j in range(dim1):
            for ip in range(dim0):
                for jp in range(dim1):
                    if pt_subsys == 0:
                        rho[ip, j, i, jp] = bipartitepurestate_tensor[i, j] * np.conj(bipartitepurestate_tensor[ip, jp])
                    else:
                        rho[i, jp, ip, j] = bipartitepurestate_tensor[i, j] * np.conj(bipartitepurestate_tensor[ip, jp])

    return rho

def bipartitepurestate_partialtranspose_subsys0_densitymatrix_cython(np.ndarray bipartitepurestate_tensor):
    cdef int i, j, ip, jp
    cdef int dim0 = bipartitepurestate_tensor.shape[0]
    cdef int dim1 = bipartitepurestate_tensor.shape[1]

    cdef np.ndarray rho = np.zeros((dim0, dim1, dim0, dim1), dtype=np.complex)

    for i in range(dim0):
        for j in range(dim1):
            for ip in range(dim0):
                for jp in range(dim1):
                    rho[ip, j, i, jp] = bipartitepurestate_tensor[i, j] * np.conj(bipartitepurestate_tensor[ip, jp])

    return rho



def bipartitepurestate_partialtranspose_subsys1_densitymatrix_cython(np.ndarray bipartitepurestate_tensor):
    cdef int i, j, ip, jp
    cdef int dim0 = bipartitepurestate_tensor.shape[0]
    cdef int dim1 = bipartitepurestate_tensor.shape[1]

    cdef np.ndarray rho = np.zeros((dim0, dim1, dim0, dim1), dtype=np.complex)

    for i in range(dim0):
        for j in range(dim1):
            for ip in range(dim0):
                for jp in range(dim1):
                    rho[i, jp, ip, j] = bipartitepurestate_tensor[i, j] * np.conj(bipartitepurestate_tensor[ip, jp])

    return rho
