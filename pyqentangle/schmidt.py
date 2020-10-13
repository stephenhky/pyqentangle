
import numpy as np
import tensornetwork as tn


def schmidt_decomposition_numpy(bipartitepurestate_tensor):
    """ Calculate the Schmidt decomposition of the given discrete bipartite quantum system

    This is called by :func:`schmidt_decomposition`. This runs numpy.

    :param bipartitepurestate_tensor: tensor describing the bi-partitite states, with each elements the coefficients for :math:`|ij\\rangle`
    :return: list of tuples containing the Schmidt coefficient, eigenmode for first subsystem, and eigenmode for second subsystem
    :type bipartitepurestate_tensor: numpy.ndarray
    :rtype: list
    """
    state_dims = bipartitepurestate_tensor.shape
    mindim = np.min(state_dims)

    vecs1, diags, vecs2_h = np.linalg.svd(bipartitepurestate_tensor)
    vecs2 = vecs2_h.transpose()

    decomposition = [(diags[k], vecs1[:, k], vecs2[:, k])
                     for k in range(mindim)]

    decomposition = sorted(decomposition, key=lambda dec: dec[0], reverse=True)

    return decomposition


def schmidt_decomposition_tensornetwork(bipartitepurestate_tensor):
    """ Calculate the Schmidt decomposition of the given discrete bipartite quantum system

    This is called by :func:`schmidt_decomposition`. This runs tensornetwork.

    :param bipartitepurestate_tensor: tensor describing the bi-partitite states, with each elements the coefficients for :math:`|ij\\rangle`
    :return: list of tuples containing the Schmidt coefficient, eigenmode for first subsystem, and eigenmode for second subsystem
    :type bipartitepurestate_tensor: numpy.ndarray
    :rtype: list
    """
    state_dims = bipartitepurestate_tensor.shape
    mindim = np.min(state_dims)

    node = tn.Node(bipartitepurestate_tensor)
    vecs1, diags, vecs2_h, _ = tn.split_node_full_svd(node, [node[0]], [node[1]])

    decomposition = [(diags.tensor[k, k], vecs1.tensor[:, k], vecs2_h.tensor[k, :])
                     for k in range(mindim)]

    decomposition = sorted(decomposition, key=lambda dec: dec[0], reverse=True)

    return decomposition


def schmidt_decomposition(bipartitepurestate_tensor, approach='tensornetwork'):
    """Calculate the Schmidt decomposition of the given discrete bipartite quantum system

    Given a discrete normalized quantum system, given in terms of 2-D numpy array ``bipartitepurestate_tensor``,
    each element of ``bipartitepurestate_tensor[i, j]`` is the coefficient of the ket :math:`|ij\\rangle`,
    calculate its Schmidt decomposition, returned as a list of tuples, where each tuple contains
    the Schmidt coefficient, the vector of eigenmode of first subsystem, and the vector of the eigenmode of
    second subsystem.

    :param bipartitepurestate_tensor: tensor describing the bi-partitite states, with each elements the coefficients for :math:`|ij\\rangle`
    :param approach: using `numpy` or `tensornetwork` in computation. (default: `tensornetwork`)
    :return: list of tuples containing the Schmidt coefficient, eigenmode for first subsystem, and eigenmode for second subsystem
    :type bipartitepurestate_tensor: numpy.ndarray
    :type approach: str
    :rtype: list
    :raise: ValueError
    """
    if approach == 'numpy':
        return schmidt_decomposition_numpy(bipartitepurestate_tensor)
    elif approach == 'tensornetwork':
        return schmidt_decomposition_tensornetwork(bipartitepurestate_tensor)
    else:
        raise ValueError("Approach is either 'numpy' or 'tensorflow', not {}.".format(approach))
