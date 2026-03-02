
import numpy as np
from scipy.stats import norm
import pytest

from pyqentangle.core.wavefunctions import Analytic1DWaveFunction


def test_gaussian_wavefunction():
    gaussian_wavefunction = Analytic1DWaveFunction(
        norm.pdf,
        to_vectorize=False
    )

    assert gaussian_wavefunction(0) == pytest.approx(0.3989422804014327)

