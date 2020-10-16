
import numpy as np
import tensornetwork as tn

from . import InvalidQuantumStateException
from .tncompute import bipartitepurestate_partialtranspose_densitymatrix, flatten_bipartite_densitymatrix


def schmidt_coefficients(schmidt_modes):
    """ Retrieving Schmidt coefficients from Schmidt modes.

    :param schmidt_modes: Schmidt modes
    :return: Schmidt coefficients
    :type schmidt_modes: list
    :rtype: numpy.array
    """
    return np.array([mode[0] for mode in schmidt_modes])


def entanglement_entropy(schmidt_modes):
    """ Calculate the entanglement entropy

    Given the calculated Schmidt modes, compute the entanglement entropy
    with the formula :math:`H=-\\sum_i p_i \log p_i`.

    :param schmidt_modes: Schmidt modes
    :return: the entanglement entropy
    :type schmidt_modes: list
    :rtype: numpy.float

    """
    eigenvalues = np.real(schmidt_coefficients(schmidt_modes))
    square_eigenvalues = np.square(np.extract(eigenvalues > 0, eigenvalues))
    entropy = np.sum(- square_eigenvalues * np.log(square_eigenvalues))
    return entropy


# participation ratio
def participation_ratio(schmidt_modes):
    """ Calculate the participation ratio

    Given the calculated Schmidt modes, compute the participation ratio
    with the formula :math:`K=\\frac{1}{\\sum_i p_i^2}`.

    :param schmidt_modes: Schmidt modes
    :return: participation ratio
    :type schmidt_modes: list
    :rtype: numpy.float

    """
    eigenvalues = np.real(np.real(schmidt_coefficients(schmidt_modes)))
    K = 1. / np.sum(np.square(np.square(eigenvalues)))
    return K


# negativity
def negativity(bipartite_tensor):
    """ Calculate the negativity

    Given a normalized bipartite discrete state, compute the negativity
    with the formula :math:`N = \\frac{||\\rho^{\Gamma_A}||_1-1}{2}`

    :param bipartite_tensor: tensor describing the bi-partitite states, with each elements the coefficients for :math:`|ij\\rangle`
    :return: negativity
    :type bipartite_tensor: numpy.ndarray
    :rtype: numpy.float

    """
    dim0, dim1 = bipartite_tensor.shape
    flatten_fullden_pt = flatten_bipartite_densitymatrix(
        bipartitepurestate_partialtranspose_densitymatrix(
            bipartite_tensor,
            0 if dim0<dim1 else 1
        )
    )

    eigenvalues = np.linalg.eigvals(flatten_fullden_pt)
    return 0.5 * (np.sum(np.abs(eigenvalues)) - 1)


# concurrence
def concurrence(bipartite_tensor):
    """ Calculate the concurrence of a bipartite system that contains 2-dimensional state qubit only.

    :param bipartite_tensor: tensor describing the bi-partitite states, with each elements the coefficients for :math:`|ij\\rangle`
    :return: concurrence
    :type bipartite_tensor: numpy.ndarray
    :rtype: numpy.float
    """
    dim0, dim1 = bipartite_tensor.shape
    if dim0 != 2 and dim1 != 2:
        raise InvalidQuantumStateException('Both or one of the subsystems have more than one bases.')

    # Levi-Civita symbol
    epsilon = np.array([[0., 1.], [-1., 0.]])

    # defining tensorflow node
    psi_node = tn.Node(bipartite_tensor, name='psi')
    # psiprime_node = tn.Node(np.conj(bipartite_tensor, name='psiprime'))
    psiprime_node = tn.Node(bipartite_tensor, name='psiprime')
    eps1_node = tn.Node(epsilon, name='epsilon1')
    eps2_node = tn.Node(epsilon, name='epsilon2')

    # defining edges for contraction
    edges = [psi_node[0] ^ eps1_node[0],
             eps1_node[1] ^ psiprime_node[0],
             psiprime_node[1] ^ eps2_node[1],
             eps2_node[0] ^ psi_node[1]]

    # computation by contraction
    t = None
    for edge in edges:
        t = tn.contract(edge)

    # concurrence
    return np.abs(t.tensor)