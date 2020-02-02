
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


def bipartitepurestate_partialtranspose_densitymatrix(bipartite_tensor, pt_subsys):
    """ Calculate the partial transpose of a density matrix.

    :param bipartite_tensor: matrix for the bipartite system
    :param pt_subsys: subsystem to transpose (either 0 or 1)
    :return: density matrix after partial transpose
    :type bipartite_tensor: numpy.ndarray
    :type pt_subsys: int
    :rtype: numpy.ndarray
    """
    if not (pt_subsys in [0, 1]):
        raise ValueError('pt_subsys can only be 0 or 1!')

    ketnode = tn.Node(bipartite_tensor)
    branode = tn.Node(np.conj(bipartite_tensor))
    final_node = tn.outer_product(ketnode, branode)

    e0, e1, e2, e3 = final_node[0], final_node[1], final_node[2], final_node[3]
    if pt_subsys == 0:
        final_node.reorder_edges([e2, e1, e0, e3])
    else:
        final_node.reorder_edges([e0, e3, e2, e1])

    return final_node.tensor


def flatten_bipartite_densitymatrix(bipartite_tensor):
    """ Flatten a bipartite state density matrix to a rank-2 matrix.

    :param bipartite_tensor: density matrix (rank-4) for the bipartite system
    :return: flatten rank-2 density matrix
    :type bipartite_tensor: numpy.ndarray
    :rtype: numpy.ndarray
    """
    denmat_node = tn.Node(bipartite_tensor)
    e0, e1, e2, e3 = denmat_node[0], denmat_node[1], denmat_node[2], denmat_node[3]
    tn.flatten_edges([e0, e1])
    tn.flatten_edges([e2, e3])
    return denmat_node.tensor