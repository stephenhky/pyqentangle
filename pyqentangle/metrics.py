
import numpy as np


def schmidt_coefficients(schmidt_modes):
    """ Retrieving Schmidt coefficients from Schmidt modes.

    :param schmidt_modes: Schmidt modes
    :return: Schmidt coefficients
    :type schmidt_modes: list
    :rtype: numpy.array
    """
    return np.array(list(map(lambda item: item[0], schmidt_modes)))


def entanglement_entropy(schmidt_modes):
    """Calculate the entanglement entropy

    Given the calculated Schmidt modes, compute the entanglement entropy
    with the formula :math:`H=-\\sum_i p_i \log p_i`.

    :param schmidt_modes: Schmidt modes
    :return: the entanglement entropy
    :type schmidt_modes: list
    :rtype: numpy.float

    """
    eigenvalues = np.real(schmidt_coefficients(schmidt_modes))
    eigenvalues = np.extract(eigenvalues > 0, eigenvalues)
    entropy = np.sum(- eigenvalues * np.log(eigenvalues))
    return entropy


# participation ratio
def participation_ratio(schmidt_modes):
    """Calculate the participation ratio

    Given the calculated Schmidt modes, compute the participation ratio
    with the formula :math:`K=\\frac{1}{\\sum_i p_i^2}`.

    :param schmidt_modes: Schmidt modes
    :return: participation ratio
    :type schmidt_modes: list
    :rtype: numpy.float

    """
    eigenvalues = np.real(np.real(schmidt_coefficients(schmidt_modes)))
    K = 1. / np.sum(eigenvalues * eigenvalues)
    return K


# negativity
def negativity(schmidt_modes):
    """Calculate the negativity

    Given the calculated Schmidt modes, compute the negativity
    with the formula :math:`N = \\frac{||\\rho||_1-1}{2}`

    :param schmidt_modes: Schmidt modes
    :return: negativity
    :type schmidt_modes: list
    :rtype: numpy.float

    """
    eigenvalues = np.real(np.real(schmidt_coefficients(schmidt_modes)))
    return 0.5 * (np.sum(np.abs(eigenvalues)) - 1)
