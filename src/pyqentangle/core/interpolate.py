
import numba as nb
import numpy as np
import numpy.typing as npt
from deprecation import deprecated

from .exceptions import UnequalLengthException, OutOfRangeException


@nb.njit(nb.complex128(nb.float64[:], nb.complex128[:], nb.float64))
def interpolate(
        xarray: npt.NDArray[np.float64],
        yarray: npt.NDArray[np.complex128],
        x: float
) -> np.complex128:
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


def numerical_continuous_interpolation(
        xarray: npt.NDArray[np.float64],
        yarray: npt.NDArray[np.complex128],
        x: float
) -> np.complex128:
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


@deprecated(deprecated_in="5.0.0", removed_in="6.0.0")
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
