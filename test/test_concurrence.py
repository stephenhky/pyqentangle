
import numpy as np
import pyqentangle
import pytest


def testSinglet():
    singlet = np.array([[0., 1.], [1.j, 0.]]) / np.sqrt(2)
    assert pyqentangle.concurrence(singlet) == pytest.approx(1., abs=1e-6)
