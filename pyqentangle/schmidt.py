from itertools import product

import numpy as np
from numpy.linalg import eig


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
    state_dims = bipartitepurestate_tensor.shape
    rho = np.zeros(state_dims * 2, dtype=np.complex)
    for i, j, ip, jp in product(*map(range, state_dims * 2)):
        rho[i, j, ip, jp] = bipartitepurestate_tensor[i, j] * np.conj(bipartitepurestate_tensor[ip, jp])
    return rho


def bipartitepurestate_reduceddensitymatrix(bipartitepurestate_tensor, kept):
    """Calculate the reduced density matrix for the specified subsystem

    Given a discrete normalized quantum system, given in terms of 2-D numpy array ``bipartitepurestate_tensor``,
    each element of ``bipartitepurestate_tensor[i, j]`` is the coefficient of the ket :math:`|ij\\rangle`,
    calculate the reduced density matrix of the specified subsystem.

    :param bipartitepurestate_tensor: tensor describing the bi-partitite states, with each elements the coefficients for :math:`|ij\\rangle`
    :param kept: subsystem, 0 indicating the first subsystem; 1 the second
    :return: reduced density matrix of the specified subsystem
    :type bipartitepurestate_tensor: numpy.ndarray
    :type kept: int
    :rtype: numpy.ndarray

    """
    state_dims = bipartitepurestate_tensor.shape
    if not (kept in [0, 1]):
        raise ValueError('kept can only be 0 or 1!')
    rho = np.zeros((state_dims[kept],) * 2, dtype=np.complex)
    for i, ip in product(*map(range, (state_dims[kept],) * 2)):
        if kept == 0:
            rho[i, ip] = np.sum([bipartitepurestate_tensor[i, j] * np.conj(bipartitepurestate_tensor[ip, j])
                                 for j in range(state_dims[1])])
        else:
            rho[i, ip] = np.sum([bipartitepurestate_tensor[j, i] * np.conj(bipartitepurestate_tensor[j, ip])
                                 for j in range(state_dims[0])])
    return rho


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

    rho1 = bipartitepurestate_reduceddensitymatrix(bipartitepurestate_tensor, 1)
    eigenvalues1, unitarymat1 = eig(rho1)
    inv_unitarymat1 = np.linalg.inv(unitarymat1)
    coefmat0 = np.dot(bipartitepurestate_tensor, inv_unitarymat1)

    decomposition = [(float(np.real(eigenvalues1[k])),
                      coefmat0[:, k] / np.sqrt(np.real(eigenvalues1[k])),
                      unitarymat1[:, k])
                     for k in range(mindim)]
    decomposition = sorted(decomposition, key=lambda dec: dec[0], reverse=True)

    return decomposition
