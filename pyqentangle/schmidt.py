
import numpy as np
import tensornetwork as tn


# total density matrix
def bipartitepurestate_densitymatrix(bipartitepurestate_tensor):
    """Calculate the whole density matrix of the bipartitite system

    Given a discrete normalized quantum system, given in terms of 2-D numpy array ``bipartitepurestate_tensor``,
    each element of ``bipartitepurestate_tensor[i, j]`` is the coefficient of the ket :math:`|ij\\rangle`,
    calculate the whole density matrix.

    :param bipartitepurestate_tensor: tensor describing the bi-partitite states, with each elements the coefficients for :math:`|ij\\rangle`
    :return: density matrix
    :type bipartitepurestate_tensor: numpy.ndarray
    :rtype: numpy.ndarray
    """
    ketnode = tn.Node(bipartitepurestate_tensor)
    branode = tn.Node(np.conj(bipartitepurestate_tensor))
    denmat_node = tn.outer_product(ketnode, branode)
    return denmat_node.tensor


def bipartitepurestate_reduceddensitymatrix(bipartitepurestate_tensor, kept):
    """Calculate the reduced density matrix for the specified subsystem

    Given a discrete normalized quantum system, given in terms of 2-D numpy array ``bipartitepurestate_tensor``,
    each element of ``bipartitepurestate_tensor[i, j]`` is the coefficient of the ket :math:`|ij\\rangle`,
    calculate the reduced density matrix of the specified subsystem.

    :param bipartitepurestate_tensor: tensor describing the bi-partitite states, with each elements the coefficients for :math:`|ij\\rangle`
    :param kept: subsystem, 0 indicating the first subsystem; 1 the second
    :param use_cython: use legacy Cython code (default: False)
    :return: reduced density matrix of the specified subsystem
    :type bipartitepurestate_tensor: numpy.ndarray
    :type kept: int
    :type use_cython: bool
    :rtype: numpy.ndarray

    """
    if not (kept in [0, 1]):
        raise ValueError('kept can only be 0 or 1!')

    ketnode = tn.Node(bipartitepurestate_tensor)
    branode = tn.Node(np.conj(bipartitepurestate_tensor))

    _ = ketnode[1-kept] ^ branode[1-kept]   # defining the edge
    reddenmat_node = ketnode @ branode     # contract

    return reddenmat_node.tensor


def schmidt_decomposition(bipartitepurestate_tensor):
    """Calculate the Schmidt decomposition of the given discrete bipartite quantum system

    Given a discrete normalized quantum system, given in terms of 2-D numpy array ``bipartitepurestate_tensor``,
    each element of ``bipartitepurestate_tensor[i, j]`` is the coefficient of the ket :math:`|ij\\rangle`,
    calculate its Schmidt decomposition, returned as a list of tuples, where each tuple contains
    the Schmidt coefficient, the vector of eigenmode of first subsystem, and the vector of the eigenmode of
    second subsystem.

    :param bipartitepurestate_tensor: tensor describing the bi-partitite states, with each elements the coefficients for :math:`|ij\\rangle`
    :return: list of tuples containing the Schmidt coefficient, eigenmode for first subsystem, and eigenmode for second subsystem
    :type bipartitepurestate_tensor: numpy.ndarray
    :rtype: list

    """
    state_dims = bipartitepurestate_tensor.shape
    mindim = np.min(state_dims)

    vecs1, diags, vecs2_h = np.linalg.svd(bipartitepurestate_tensor)
    vecs2 = vecs2_h.transpose()
    #decomposition = [(diags[k]*diags[k], vecs1[:, k], transposevecs2[:, k])
    #                 for k in range(mindim)]
    decomposition = [(diags[k], vecs1[:, k], vecs2[:, k])
                     for k in range(mindim)]

    decomposition = sorted(decomposition, key=lambda dec: dec[0], reverse=True)

    return decomposition
