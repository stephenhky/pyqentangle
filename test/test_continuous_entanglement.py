
import numpy as np
from scipy.integrate import quad
import pytest

from pyqentangle.core.wavefunctions import AnalyticMultiDimWaveFunction
from pyqentangle.entangle import ContinuousSchmidtDecomposer


normsq = lambda x: x*np.conj(x)


def test_entangled_oscillators():
    wavefcn = AnalyticMultiDimWaveFunction(
        lambda x: np.exp(-0.5 * (x[0] + x[1]) ** 2) * np.exp(-(x[0] - x[1]) ** 2) * np.sqrt(np.sqrt(8.) / np.pi)
    )
    rho = 3 - 2*np.sqrt(2)
    a = np.sqrt((1-rho*rho)/(2*rho))
    expected_coef = lambda n: np.sqrt(np.sqrt(8)*(1-rho*rho))/a*rho**n

    for approach in ['tensornetwork', 'numpy']:
        print(f"--- Approach: {approach} ----")
        decompositions = ContinuousSchmidtDecomposer(
            wavefcn, -10., 10., -10., 10., keep=10, approach=approach
        ).modes()
        for i in range(10):
            print(f'mode {i}:  expected={expected_coef(i)}, calculated={decompositions[i].schmidt_coef}')
            assert expected_coef(i) == pytest.approx(decompositions[i].schmidt_coef)

        schmidt_fcn1 = decompositions[0].wavefunction1
        norm1, err1 = quad(
            lambda x1: np.real(normsq(schmidt_fcn1(np.array([x1]))))[0],
            -10,
            10
        )
        schmidt_fcn2 = decompositions[0].wavefunction2
        norm2, err2 = quad(
            lambda x2: np.real(normsq(schmidt_fcn2(np.array([x2]))))[0],
            -10,
            10
        )
        assert norm1 == pytest.approx(1., abs=1e-2)
        assert norm2 == pytest.approx(1., abs=1e-2)
