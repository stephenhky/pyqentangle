import numpy as np
from numpy.linalg import eigvals


def entanglement_entropy(reduceddensitymatrix):
    """Calculate the entanglement entropy

    Given the reduced density matrix of a subsystem, compute the entanglement entropy
    with the formula :math:`H=-\\sum_i p_i \log p_i`.

    :param reduceddensitymatrix: reduced density matrix of a subsystem
    :return: the entanglement entropy
    :type reduceddensitymatrix: numpy.ndarray
    :rtype: numpy.float

    """
    eigenvalues = eigvals(reduceddensitymatrix)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = eigenvalues[ eigenvalues > 0]
    entropy = np.sum(- eigenvalues * np.log(eigenvalues))
    return entropy


# participation ratio
def participation_ratio(reduceddensitymatrix):
    """Calculate the participation ratio

    Given the reduced density matrix of a subsystem, compute the participation ratio
    with the formula :math:`K=\\frac{1}{\\sum_i p_i^2}`.

    :param reduceddensitymatrix: reduced density matrix of a subsystem
    :return: participation ratio
    :type reduceddensitymatrix: numpy.ndarray
    :rtype: numpy.float

    """
    eigenvalues = eigvals(reduceddensitymatrix)
    eigenvalues = np.real(eigenvalues)
    K = 1. / np.sum(eigenvalues * eigenvalues)
    return K


# negativity
def negativity(reduceddensitymatrix):
    """Calculate the negativity

    Given the reduced density matrix of a subsystem, compute the negativity
    with the formula :math:`N = \\frac{||\\rho||_1-1}{2}`

    :param reduceddensitymatrix: reduced density matrix of a subsystem
    :return: negativity
    :type reduceddensitymatrix: numpy.ndarray
    :rtype: numpy.float

    """
    eigenvalues = eigvals(reduceddensitymatrix)
    return 0.5 * (np.sum(np.abs(eigenvalues)) - 1)
