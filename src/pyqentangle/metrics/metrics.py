
import warnings

import numpy as np
import numpy.typing as npt
import tensornetwork as tn

from ..core.exceptions import InvalidQuantumStateException
from ..schemas.schemas import SchmidtMode
from ..core.tncompute import bipartitepurestate_partialtranspose_densitymatrix, flatten_bipartite_densitymatrix


def schmidt_coefficients(schmidt_modes: list[SchmidtMode]) -> npt.NDArray[np.float64]:
    """Extract the Schmidt coefficients from a list of Schmidt modes.

    Args:
        schmidt_modes (list[SchmidtMode]): Schmidt modes as returned by the Schmidt
            decomposition routines.

    Returns:
        npt.NDArray[numpy.float64]: 1-D array of Schmidt coefficients, one per mode,
        in the same order as ``schmidt_modes``.
    """
    return np.array([mode.schmidt_coef for mode in schmidt_modes])


def entanglement_entropy(schmidt_modes: list[SchmidtMode]) -> float:
    """Compute the von Neumann entanglement entropy from Schmidt modes.

    Uses the formula

    .. math::

        S = -\\sum_i p_i \\log p_i

    where :math:`p_i = \\lambda_i^2` are the squared Schmidt coefficients (eigenvalues of
    the reduced density matrix).  Terms with :math:`\\lambda_i = 0` are excluded to avoid
    :math:`0 \\log 0` singularities.

    Args:
        schmidt_modes (list[SchmidtMode]): Schmidt modes as returned by the Schmidt
            decomposition routines.

    Returns:
        float: Von Neumann entanglement entropy :math:`S \\geq 0`.  Returns ``0`` for a
        product state and :math:`\\log(\\min(d_1, d_2))` for a maximally entangled state.
    """
    eigenvalues = np.real(schmidt_coefficients(schmidt_modes))
    square_eigenvalues = np.square(np.extract(eigenvalues > 0, eigenvalues))
    entropy = np.sum(- square_eigenvalues * np.log(square_eigenvalues))
    return entropy


# Renyi's entropy
def renyi_entanglement_entropy(schmidt_modes: list[SchmidtMode], alpha: float) -> float:
    """Compute the Rényi entanglement entropy of order ``alpha`` from Schmidt modes.

    Uses the formula

    .. math::

        S_\\alpha = \\frac{1}{1-\\alpha} \\log \\sum_i p_i^{\\alpha}

    where :math:`p_i = \\lambda_i^2` are the squared Schmidt coefficients.  In the limit
    :math:`\\alpha \\to 1` this reduces to the von Neumann entropy; when ``alpha=1`` is
    passed, the function falls back to :func:`entanglement_entropy` and emits a warning.

    Args:
        schmidt_modes (list[SchmidtMode]): Schmidt modes as returned by the Schmidt
            decomposition routines.
        alpha (float): Rényi order parameter.  Must satisfy :math:`\\alpha \\geq 0` and
            :math:`\\alpha \\neq 1` (use ``alpha=1`` to obtain the von Neumann entropy via
            the fallback path).

    Returns:
        float: Rényi entanglement entropy :math:`S_\\alpha`.
    """
    if alpha == 1:
        warnings.warn('alpha = 1, doing Shannon entanglement entropy.')
        return entanglement_entropy(schmidt_modes)
    eigenvalues = np.real(schmidt_coefficients(schmidt_modes))
    square_eigenvalues = np.square(np.extract(eigenvalues > 0, eigenvalues))
    renyi_entropy = np.log(np.sum(square_eigenvalues**alpha)) / (1-alpha)
    return renyi_entropy


# participation ratio
def participation_ratio(schmidt_modes: list[SchmidtMode]) -> float:
    """Compute the participation ratio (Schmidt number) from Schmidt modes.

    Uses the formula

    .. math::

        K = \\frac{1}{\\sum_i p_i^2}

    where :math:`p_i = \\lambda_i^2` are the squared Schmidt coefficients.  :math:`K`
    equals 1 for a product state and :math:`\\min(d_1, d_2)` for a maximally entangled
    state, providing an effective count of the contributing Schmidt modes.

    Args:
        schmidt_modes (list[SchmidtMode]): Schmidt modes as returned by the Schmidt
            decomposition routines.

    Returns:
        float: Participation ratio :math:`K \\geq 1`.
    """
    eigenvalues = np.real(np.real(schmidt_coefficients(schmidt_modes)))
    K = 1. / np.sum(np.square(np.square(eigenvalues)))
    return K


# negativity
def negativity(bipartite_tensor: npt.NDArray[np.complex128]) -> float:
    """Compute the negativity of a discrete bipartite pure state.

    The negativity is defined as

    .. math::

        N(\\rho) = \\frac{\\|\\rho^{\\Gamma_A}\\|_1 - 1}{2}

    where :math:`\\rho^{\\Gamma_A}` is the partial transpose of the density matrix with
    respect to the smaller subsystem and :math:`\\|\\cdot\\|_1` denotes the trace norm
    (sum of absolute eigenvalues).  A non-zero negativity certifies entanglement.

    The partial transpose is taken over the subsystem with the smaller dimension
    (``0`` if ``d1 < d2``, otherwise ``1``).

    Args:
        bipartite_tensor (npt.NDArray[numpy.complex128]): 2-D complex array of shape
            ``(d1, d2)`` representing a normalised bipartite pure state, where element
            ``[i, j]`` is the coefficient of the basis ket :math:`|ij\\rangle`.

    Returns:
        float: Negativity :math:`N \\geq 0`.  Returns ``0`` for a separable state.
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
def concurrence(bipartite_tensor: npt.NDArray[np.complex128]) -> float:
    """Compute the concurrence of a two-qubit bipartite pure state.

    The concurrence is an entanglement measure defined for systems where at least one
    subsystem is a qubit (dimension 2).  It is computed via the tensor-network contraction

    .. math::

        C = |\\langle\\Psi| (\\epsilon \\otimes \\epsilon) |\\Psi^*\\rangle|

    where :math:`\\epsilon = \\begin{pmatrix} 0 & 1 \\\\ -1 & 0 \\end{pmatrix}` is the
    Levi-Civita symbol.  :math:`C = 0` for a product state and :math:`C = 1` for a
    maximally entangled Bell state.

    Args:
        bipartite_tensor (npt.NDArray[numpy.complex128]): 2-D complex array of shape
            ``(d1, d2)`` representing a normalised bipartite pure state, where element
            ``[i, j]`` is the coefficient of the basis ket :math:`|ij\\rangle`.
            At least one of ``d1``, ``d2`` must equal ``2``.

    Returns:
        float: Concurrence :math:`C \\in [0, 1]`.

    Raises:
        InvalidQuantumStateException: If neither subsystem has dimension 2.
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
