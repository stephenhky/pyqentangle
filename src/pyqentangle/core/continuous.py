
from itertools import product
from typing import Optional, Literal

import numpy as np

from .schmidt import schmidt_decomposition
from .interpolate import numerical_continuous_function
from .wavefunctions import InterpolatingWaveFunction, WaveFunction


def discretize_continuous_bipartitesys(
        fcn: callable,
        x1_lo: float,
        x1_hi: float,
        x2_lo: float,
        x2_hi: float,
        nb_x1: int = 100,
        nb_x2: int = 100
) -> np.ndarray:
    """Find the discretized representation of the continuous bipartite system.

    Given a function `fcn` (a function with two input variables),
    find the discretized representation of the bipartite system, with
    the first system ranges from `x1_lo` to `x1_hi`, and second from `x2_lo` to `x2_hi`.

    Args:
        fcn (function): Function with two input variables.
        x1_lo (float): Lower bound of :math:`x_1`.
        x1_hi (float): Upper bound of :math:`x_1`.
        x2_lo (float): Lower bound of :math:`x_2`.
        x2_hi (float): Upper bound of :math:`x_2`.
        nb_x1 (int, optional): Number of :math:`x_1`. Defaults to 100.
        nb_x2 (int, optional): Number of :math:`x_2`. Defaults to 100.

    Returns:
        numpy.ndarray: Discretized tensor representation of the continuous bipartite system.
    """
    x1 = np.linspace(x1_lo, x1_hi, nb_x1)
    x2 = np.linspace(x2_lo, x2_hi, nb_x2)
    tensor = np.zeros((len(x1), len(x2)), dtype=np.complex128)
    for i, j in product(*map(range, tensor.shape)):
        tensor[i, j] = fcn(x1[i], x2[j])
    return tensor


def continuous_schmidt_decomposition(
        fcn: callable,
        x1_lo: float,
        x1_hi: float,
        x2_lo: float,
        x2_hi: float,
        nb_x1: int = 100,
        nb_x2: int = 100,
        keep: Optional[int] = None,
        approach: Literal["tensornetwork", "numpy"] = 'tensornetwork'
) -> list[tuple[float, WaveFunction, WaveFunction]]:
    """Compute the Schmidt decomposition of a continuous bipartite quantum systems.

    Given a function `fcn` (a function with two input variables), perform the Schmidt
    decomposition, returning a list of tuples, where each contains a Schmidt decomposition,
    the lambda function of the eigenmode in the first subsystem, and the lambda function
    of the eigenmode of the second subsystem.

    Args:
        fcn (function): Function with two input variables.
        x1_lo (float): Lower bound of :math:`x_1`.
        x1_hi (float): Upper bound of :math:`x_1`.
        x2_lo (float): Lower bound of :math:`x_2`.
        x2_hi (float): Upper bound of :math:`x_2`.
        nb_x1 (int, optional): Number of :math:`x_1`. Defaults to 100.
        nb_x2 (int, optional): Number of :math:`x_2`. Defaults to 100.
        keep (int, optional): The number of Schmidt modes with the largest coefficients to return; 
            the smaller of `nb_x1` and `nb_x2` will be returned if `None` is given. Defaults to `None`.
        approach (str, optional): Using `numpy` or `tensornetwork` in computation. Defaults to `tensornetwork`.

    Returns:
        list: List of tuples, where each contains a Schmidt coefficient, the lambda function of the 
        eigenmode of the first subsystem, and the lambda function of the eigenmode of the second subsystem.

    Raises:
        ValueError: If approach is not 'numpy' or 'tensornetwork'.
    """
    tensor = discretize_continuous_bipartitesys(fcn, x1_lo, x1_hi, x2_lo, x2_hi, nb_x1=nb_x1, nb_x2=nb_x2)
    decomposition = schmidt_decomposition(tensor, approach=approach)

    if keep is None:
        keep = min(nb_x1, nb_x2)
    else:
        keep = len(decomposition) if keep > len(decomposition) else keep

    x1array = np.linspace(x1_lo, x1_hi, nb_x1)
    x2array = np.linspace(x2_lo, x2_hi, nb_x2)
    dx1 = (x1_hi - x1_lo) / (nb_x1 - 1.)
    dx2 = (x2_hi - x2_lo) / (nb_x2 - 1.)

    renormalized_decomposition = []
    sum_sq_eigvals = np.sum(list(map(lambda dec: dec[0]*dec[0], decomposition)))
    for i in range(keep):
        schmidt_weight, unnorm_modeA, unnorm_modeB = decomposition[i]
        normA = np.linalg.norm(unnorm_modeA) * np.sqrt(dx1)
        normB = np.linalg.norm(unnorm_modeB) * np.sqrt(dx2)
        renormalized_decomposition.append(
            (schmidt_weight / np.sqrt(sum_sq_eigvals),
             InterpolatingWaveFunction(x1array, unnorm_modeA / normA),
             InterpolatingWaveFunction(x2array, unnorm_modeB / normB)
             )
        )

    return renormalized_decomposition
