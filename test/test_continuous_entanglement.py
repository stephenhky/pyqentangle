
import numpy as np
from scipy.integrate import quad
import pytest

import pyqentangle


def test_entangled_oscillators():
    fcn = lambda x1, x2: np.exp(-0.5 * (x1 + x2) ** 2) * np.exp(-(x1 - x2) ** 2) * np.sqrt(np.sqrt(8.) / np.pi)
    rho = 3 - 2*np.sqrt(2)
    a = np.sqrt((1-rho*rho)/(2*rho))
    expected_coef = lambda n: np.sqrt(np.sqrt(8)*(1-rho*rho))/a*rho**n
    for approach in ['tensornetwork', 'numpy']:
        decompositions = pyqentangle.continuous_schmidt_decomposition(fcn, -10., 10., -10., 10., keep=10,
                                                                      approach=approach)
        eigenvalues = list(map(lambda item: item[0], decompositions))
        for i in range(10):
            print('expected={}, calculated={}'.format(expected_coef(i), eigenvalues[i]))
            assert expected_coef(i), pytest.approx(eigenvalues[i])

        norm1, err1 = quad(
            lambda x1: np.real(np.conjugate(decompositions[0][1](np.array([x1])))*decompositions[0][1](np.array([x1]))),
            -10,
            10
        )
        print(norm1)
        norm2, err2 = quad(
            lambda x2: np.real(np.conjugate(decompositions[0][2](np.array([x2])))*decompositions[0][2](np.array([x2]))),
            -10,
            10
        )
        assert norm1 == pytest.approx(1., abs=1e-2)
        assert norm2 == pytest.approx(1., abs=1e-2)


def test_interpolation():
    xarray = np.array([0., 1., 2.])
    yarray = np.array([0., 1., 4.])
    assert pyqentangle.continuous.numerical_continuous_interpolation(xarray, yarray, 0.5) == pytest.approx(0.5)
    assert pyqentangle.continuous.numerical_continuous_interpolation(xarray, yarray, 1.5) == pytest.approx(2.5)
