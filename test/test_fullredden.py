
from math import sqrt

import numpy as np

import pyqentangle
import pyqentangle.tncompute
import pytest


def test_uneven_two_levels(self):
    tensor = np.array([[0., np.sqrt(np.reciprocal(3.))],
                       [-np.sqrt(2./3.)*1j, 0.]])
    fulldenmat = pyqentangle.tncompute.bipartitepurestate_densitymatrix(tensor)

    assert fulldenmat[0, 1, 0, 1] == pytest.approx(np.reciprocal(3.))
    assert fulldenmat[1, 0, 1, 0] == pytest.approx(2/3)
    assert fulldenmat[0, 1, 1, 0] == pytest.approx(sqrt(2)/3*1j)
    assert fulldenmat[1, 0, 0, 1] == pytest.approx(-sqrt(2) / 3 * 1j)
