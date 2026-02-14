
from math import sqrt, pi

import numpy as np
from scipy.special import hermite

from ..utils import InvalidMatrix


def disentangled_gaussian_wavefcn() -> callable:
    """Return the function of normalized disentangled Gaussian systems.

    Returns:
        function: Function of two variables.
    """
    return lambda x1, x2: np.exp(-0.5 * (x1 * x1 + x2 * x2)) / np.sqrt(np.pi)


def correlated_bipartite_gaussian_wavefcn(covmatrix: np.ndarray) -> callable:
    """Return a normalized correlated bivariate Gaussian wavefunction.

    Args:
        covmatrix (numpy.array): Covariance matrix of size (2, 2).

    Returns:
        function: A wavefunction of two variables.
    """
    if not covmatrix.shape == (2, 2):
        raise InvalidMatrix("Invalid matrix shape: "+str(covmatrix.shape)+"; desired shape: (2, 2)")
    if covmatrix[0, 1] != covmatrix[1, 0]:
        raise InvalidMatrix("Not a symmetric covariance matrix")

    norm = 2 * np.pi / np.sqrt(np.linalg.det(covmatrix))
    const = 1 / np.sqrt(norm)
    return lambda x1, x2: const * np.exp(-0.25* np.matmul(np.array([[x1, x2]]),
                                                          np.matmul(covmatrix,
                                                                    np.array([[x1], [x2]])
                                                                    )
                                                          )
                                         )


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
def harmonic_wavefcn(n: int) -> callable:
    """Return the normalized wavefunction of a harmonic oscillator, where $n$ denotes
    that it is an n-th excited state, or ground state for $n=0$.

    Args:
        n (int): Quantum number of the excited state.

    Returns:
        function: A normalized wavefunction.
    """
    const = 1/sqrt(2**n * tail_factorial(n)) * 1/sqrt(sqrt(pi))
    return lambda x: const * np.exp(-0.5*x*x) * hermite(n)(x)


# excited interaction states
def coupled_excited_harmonics(n: int) -> callable:
    """Return a bipartitite wavefunction, with ground state of center of mass,
    but excited state for the interaction.

    Args:
        n (int): Quantum harmonic state number for the interaction.

    Returns:
        function: Wavefunction of two variables.
    """
    return lambda x1, x2: harmonic_wavefcn(0)(0.5*(x1+x2)) * harmonic_wavefcn(n)(x1-x2)


# tutorial on double integration: https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#general-multiple-integration-dblquad-tplquad-nquad
