
import numpy as np
from scipy.integrate import quad
import pytest

import pyqentangle
import pyqentangle.core.interpolate


def test_entangled_oscillators():
    fcn = lambda x1, x2: np.exp(-0.5 * (x1 + x2) ** 2) * np.exp(-(x1 - x2) ** 2) * np.sqrt(np.sqrt(8.) / np.pi)
    rho = 3 - 2*np.sqrt(2)
    a = np.sqrt((1-rho*rho)/(2*rho))
    expected_coef = lambda n: np.sqrt(np.sqrt(8)*(1-rho*rho))/a*rho**n
    for approach in ['tensornetwork', 'numpy']:
        print(f"--- Approach: {approach} ----")

        decompositions = pyqentangle.continuous_schmidt_decomposition(fcn, -10., 10., -10., 10., keep=10,
                                                                      approach=approach)
        eigenvalues = list(map(lambda item: item[0], decompositions))
        print(decompositions[:10])
        for i in range(10):
            print(f'mode {i}:  expected={expected_coef(i)}, calculated={eigenvalues[i]}')
            assert expected_coef(i) == pytest.approx(eigenvalues[i])

        schmidt_fcn1 = decompositions[0][1]
        norm1, err1 = quad(
            lambda x1: np.real(np.conjugate(schmidt_fcn1(np.array([x1])))*schmidt_fcn1(np.array([x1])))[0],
            -10,
            10
        )
        schmidt_fcn2 = decompositions[0][2]
        norm2, err2 = quad(
            lambda x2: np.real(np.conjugate(schmidt_fcn2(np.array([x2])))*schmidt_fcn2(np.array([x2])))[0],
            -10,
            10
        )
        assert norm1 == pytest.approx(1., abs=1e-2)
        assert norm2 == pytest.approx(1., abs=1e-2)



