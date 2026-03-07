
from math import sqrt, pi
from functools import lru_cache

import numpy as np
import numpy.typing as npt
import numba as nb
from scipy.special import hermite

from ..core.exceptions import InvalidMatrix
from ..core.wavefunctions import Analytic1DWaveFunction, AnalyticMultiDimWaveFunction, WaveFunction


@nb.njit(nb.float64(nb.float64))
def sqrt_gaussian_function_value(x: float) -> float:
    """Evaluate the square-root Gaussian function at a given point.

    Computes :math:`\\exp(-x^2/4) / (2\\pi)^{1/4}`, which is the amplitude of a
    normalized Gaussian wavefunction with unit width. This function is JIT-compiled
    with Numba for performance.

    Args:
        x (float): The coordinate at which to evaluate the function.

    Returns:
        float: The value of the square-root Gaussian at `x`.
    """
    return np.exp(-0.25*x*x) / np.sqrt(np.sqrt(2*np.pi))


def disentangled_gaussian_wavefcn() -> WaveFunction:
    """Return the function of normalized disentangled Gaussian systems.

    Returns:
        function: Function of two variables.
    """
    return AnalyticMultiDimWaveFunction(lambda x: sqrt_gaussian_function_value(x[0]) * sqrt_gaussian_function_value(x[1]))


@nb.njit(nb.float64(nb.float64[:, :], nb.float64, nb.float64))
def correlated_bipartite_gaussian_value(covmatrix: npt.NDArray[np.float64], x1: float, x2: float) -> float:
    """Evaluate a normalized correlated bivariate Gaussian wavefunction at a given point.

    Computes the amplitude of a bivariate Gaussian wavefunction with covariance matrix
    `covmatrix` at coordinates ``(x1, x2)``. The normalization ensures that the squared
    amplitude integrates to unity. This function is JIT-compiled with Numba for performance.

    Args:
        covmatrix (numpy.ndarray): Symmetric positive-definite covariance matrix of shape ``(2, 2)``.
        x1 (float): Coordinate of the first subsystem.
        x2 (float): Coordinate of the second subsystem.

    Returns:
        float: The wavefunction amplitude at ``(x1, x2)``.
    """
    norm = 2 * np.pi / np.sqrt(np.linalg.det(covmatrix))
    const = 1 / np.sqrt(norm)
    return const * np.exp(-0.25 * np.array([[x1, x2]]) @ covmatrix @ np.array([[x1], [x2]]))[0, 0]


def correlated_bipartite_gaussian_wavefcn(covmatrix: np.ndarray) -> WaveFunction:
    """Return a normalized correlated bivariate Gaussian wavefunction.

    Args:
        covmatrix (numpy.array): Covariance matrix of size (2, 2).

    Returns:
        function: A wavefunction of two variables.
    """
    if not covmatrix.shape == (2, 2):
        raise InvalidMatrix(f"Invalid matrix shape: {covmatrix.shape}; desired shape: (2, 2)")
    if covmatrix[0, 1] != covmatrix[1, 0]:
        raise InvalidMatrix("Not a symmetric covariance matrix")
    return AnalyticMultiDimWaveFunction(lambda x: correlated_bipartite_gaussian_value(covmatrix, x[0], x[1]))


@lru_cache(maxsize=100)
def tail_factorial(n: int, accumulator: int = 1) -> int:
    """Return the factorial of an integer.

    The calculation is done by tail recursion.

    Args:
        n (int): The integer of which the factorial is desired to return.
        accumulator (int, optional): Default to be 1, for tail recursion. Defaults to 1.

    Returns:
        int: Factorial of `n`.
    """
    if n == 0:
        return accumulator
    else:
        return tail_factorial(n-1, accumulator * n)


# m = omega = hbar = 1
def harmonic_wavefcn(n: int) -> WaveFunction:
    """Return the normalized wavefunction of a harmonic oscillator, where $n$ denotes
    that it is an n-th excited state, or ground state for $n=0$.

    Args:
        n (int): Quantum number of the excited state.

    Returns:
        function: A normalized wavefunction.
    """
    const = 1/sqrt(2**n * tail_factorial(n)) * 1/sqrt(sqrt(pi))
    return Analytic1DWaveFunction(lambda x: const * np.exp(-0.5*x*x) * hermite(n)(x))


# excited interaction states
def coupled_excited_harmonics(n: int) -> WaveFunction:
    """Return a bipartitite wavefunction, with ground state of center of mass,
    but excited state for the interaction.

    Args:
        n (int): Quantum harmonic state number for the interaction.

    Returns:
        function: Wavefunction of two variables.
    """
    # note: put a non-vectorize function
    return AnalyticMultiDimWaveFunction(lambda x: harmonic_wavefcn(0)(0.5*(x[0]+x[1]))[0] * harmonic_wavefcn(n)(x[0]-x[1])[0])
