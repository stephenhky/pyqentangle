
from math import sqrt, pi

import numpy as np
from scipy.special import hermite

def disentangled_gaussian_wavefcn():
    """

    :return:
    """
    return lambda x1, x2: np.exp(-0.5 * (x1 * x1 + x2 * x2)) / np.sqrt(np.pi)


def tail_factorial(n, accumulator=1):
    """

    :param n:
    :param accumulator:
    :return:
    """
    if n == 0:
        return accumulator
    else:
        return tail_factorial(n-1, accumulator * n)


# m = omega = hbar = 1
def harmonic_wavefcn(n):
    """

    :param n:
    :return:
    """
    const = 1/sqrt(2**n * tail_factorial(n)) * 1/sqrt(sqrt(pi))
    return lambda x: const * np.exp(-0.5*x*x) * hermite(n)(x)
