
from itertools import product

import numpy as np

from . import schmidt_decomposition, OutOfRangeException, UnequalLengthException


def interpolate(xarray: np.ndarray, yarray: np.ndarray, x: float) -> float:
    left = 0
    right = len(xarray) - 1
    idx = right // 2
    while (idx != 0 and idx != len(xarray) - 1) and (not (x >= xarray[idx] and x < xarray[idx + 1])):
        if x >= xarray[idx + 1]:
            left = idx + 1
        elif x < xarray[idx]:
            right = idx - 1
        idx = (left + right) // 2

    return yarray[idx] + (yarray[idx + 1] - yarray[idx]) / (xarray[idx + 1] - xarray[idx]) * (x - xarray[idx])


def numerical_continuous_interpolation(xarray: np.ndarray, yarray: np.ndarray, x: float) -> float:
    """Evaluate the value of a function given a variable x using interpolation.

    With a function approximated by given arrays of independent variable (`xarray`)
    and of dependent variable (`yarray`), the value of this function given `x` is
    calculated by interpolation.

    If `x` is outside the range of `xarray`, an `OutOfRangeException`
    is raised; if the lengths of `xarray` and `yarray` are not equal, an
    `UnequalLengthException` is raised.

    Args:
        xarray (numpy.ndarray): An array of independent variable values.
        yarray (numpy.ndarray): An array of dependent variable values.
        x (float): The input value at which the function is computed.

    Returns:
        float: The value of function with the given `x`.

    Raises:
        OutOfRangeException: If `x` is outside the range of `xarray`.
        UnequalLengthException: If the lengths of `xarray` and `yarray` are not equal.
    """
    if len(xarray) != len(yarray):
        raise UnequalLengthException(xarray, yarray)
    minx = np.min(xarray)
    maxx = np.max(xarray)
    if x == maxx:
        return yarray[-1]
    if not (x >= minx and x < maxx):
        raise OutOfRangeException(x)

    return interpolate(xarray, yarray, x)


def numerical_continuous_function(xarray: np.ndarray, yarray: np.ndarray) -> callable:
    """Return a function with the given arrays of independent and dependent variables.

    With a function approximated by given arrays of independent variable (`xarray`)
    and of dependent variable (`yarray`), it returns a lambda function that takes
    a `numpy.ndarray` as an input and calculates the values at all these elements
    using interpolation.

    If `x` is outside the range of `xarray`, an `OutOfRangeException`
    is raised.

    Args:
        xarray (numpy.ndarray): An array of independent variable values.
        yarray (numpy.ndarray): An array of dependent variable values.

    Returns:
        function: A lambda function that takes a `numpy.ndarray` as the input parameter and calculates the values.

    Raises:
        OutOfRangeException: If `x` is outside the range of `xarray`.
    """
    return lambda xs: np.array(list(map(lambda x: numerical_continuous_interpolation(xarray, yarray, x), xs)),
                               dtype=np.complex128)


def discretize_continuous_bipartitesys(fcn: callable, x1_lo: float, x1_hi: float, x2_lo: float, x2_hi: float, nb_x1: int = 100, nb_x2: int = 100) -> np.ndarray:
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


def continuous_schmidt_decomposition(fcn: callable, x1_lo: float, x1_hi: float, x2_lo: float, x2_hi: float, nb_x1: int = 100, nb_x2: int = 100, keep: int = None,
                                     approach: str = 'tensornetwork') -> list:
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
            ( schmidt_weight / np.sqrt(sum_sq_eigvals),
              numerical_continuous_function(x1array, unnorm_modeA / normA),
              numerical_continuous_function(x2array, unnorm_modeB / normB)
             )
        )

    return renormalized_decomposition
