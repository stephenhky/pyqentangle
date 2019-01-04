
import numpy as np
cimport numpy as np

def bipartitepurestate_densitymatrix_cython(np.ndarray bipartitepurestate_tensor):
    cdef int i, j, ip, jp
    cdef int dim0 = bipartitepurestate_tensor.shape[0]
    cdef int dim1 = bipartitepurestate_tensor.shape[1]

    cdef np.ndarray rho = np.zeros((dim0, dim1, dim0, dim1), dtype=np.complex)

    for i in range(dim0):
        for j in range(dim1):
            for ip in range(dim0):
                for jp in range(dim1):
                    rho[i, j, ip, jp] = bipartitepurestate_tensor[i, j] * np.conj(bipartitepurestate_tensor[ip, jp])

    return rho


def flatten_bipartite_densitymatrix_cython(np.ndarray full_denmat):
    cdef int dim0 = full_denmat.shape[0]
    cdef int dim1 = full_denmat.shape[1]
    # the other two dimensions have the same lengths

    cdef np.ndarray flatten_denmat = np.zeros((dim0*dim1, dim0*dim1), dtype=np.complex)

    for i in range(dim0):
        for j in range(dim1):
            for ip in range(dim0):
                for jp in range(dim1):
                    flatten_denmat[i*dim0+j, ip*dim0+jp] = full_denmat[i, j, ip, jp]

    return flatten_denmat
