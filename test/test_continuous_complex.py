
import numpy as np
import pytest

from pyqentangle import continuous_schmidt_decomposition
from pyqentangle.core.wavefunctions import AnalyticMultiDimWaveFunction
from pyqentangle.entangle import ContinuousSchmidtDecomposer


def test_samewavefunction_different_phases():
    f1 = lambda x1, x2: np.exp(-0.5 * (x1 + x2) ** 2) * np.exp(-(x1 - x2) ** 2) * np.sqrt(np.sqrt(8.) / np.pi)
    f2 = lambda x1, x2: f1(x1, x2) * np.sqrt(0.5) * (1 + 1j)

    f1_modes = continuous_schmidt_decomposition(f1, -10, 10, -10, 10, keep=5)
    f2_modes = continuous_schmidt_decomposition(f2, -10, 10, -10, 10, keep=5)

    for f1_mode, f2_mode in zip(f1_modes, f2_modes):
        f1_eigval = f1_mode[0]
        f2_eigval = f2_mode[0]
        np.testing.assert_almost_equal(f1_eigval, f2_eigval)


def test_samewavefunction_different_phases_new():
    wavefunction1 = AnalyticMultiDimWaveFunction(lambda x1, x2: np.exp(-0.5 * (x1 + x2) ** 2) * np.exp(-(x1 - x2) ** 2) * np.sqrt(np.sqrt(8.) / np.pi))
    wavefunction2 = np.sqrt(0.5) * (1 + 1j) * wavefunction1

    schmidt_modes_1 = ContinuousSchmidtDecomposer(
        wavefunction1, -10, 10, -10, 10, keep=5
    ).modes()
    schmidt_modes_2 = ContinuousSchmidtDecomposer(
        wavefunction2, -10, 10, -10, 10, keep=5
    ).modes()

    for schmidt_mode_1, schmidt_mode_2 in zip(schmidt_modes_1, schmidt_modes_2):
        assert schmidt_mode_1.schmidt_coef == pytest.approx(schmidt_mode_2.schmidt_coef)

