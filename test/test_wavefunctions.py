
from math import sqrt, pi, cos, sin

import numpy as np
from scipy.stats import norm
from scipy.special import hermite
import pytest

from pyqentangle.core.wavefunctions import Analytic1DWaveFunction


def test_gaussian_wavefunction():
    gaussian_wavefunction = Analytic1DWaveFunction(
        norm.pdf,
        to_vectorize=False
    )
    assert gaussian_wavefunction(0) == pytest.approx(norm.pdf(0))
    assert gaussian_wavefunction.prob_density(0) == pytest.approx(np.square(norm.pdf(0)))

    half_gaussian_wavefunction = 0.5 * gaussian_wavefunction
    assert half_gaussian_wavefunction(0) == pytest.approx(0.5 * norm.pdf(0))


def test_comb_2mode_harmonic_wavefunction():
    psi0 = Analytic1DWaveFunction(
        lambda x: 1. / sqrt(sqrt(pi)) * np.exp(-0.5*x*x) * hermite(0)(x),
        to_vectorize=False
    )
    psi1 = Analytic1DWaveFunction(
        lambda x: sqrt(1. / (2. * sqrt(pi))) * np.exp(-0.5*x*x) * hermite(1)(x),
        to_vectorize=False
    )

    comb_fcn = sqrt(0.5) * psi0 + sqrt(0.5) * (np.cos(0.25*pi)+np.sin(0.25*pi)*1j) * psi1

    assert comb_fcn(0) == pytest.approx(sqrt(0.5) / sqrt(sqrt(pi)))
