
from math import log

import numpy as np
import pytest

import pyqentangle


tensor = np.array([[0., np.sqrt(0.7)], [np.sqrt(0.3), 0.]])
shannon = log(2.)
schmidt_modes = pyqentangle.DiscreteSchmidtDecomposer(tensor).modes()


def test_entanglement_entropy():
    assert pyqentangle.entanglement_entropy(schmidt_modes) == pytest.approx(0.8812908992306926*shannon)


def test_renyi_entanglement_entropies():
    alpha_to_entropies = {
        0.0: 1.0,
        0.5: 0.9384853943613469,
        1.0: 0.8812908992306926,
        1.5: 0.8301566146039395,
        2.0: 0.7858751946471525,
        2.5: 0.748414566844965,
        5.0: 0.6380390889094786,
        10.0: 0.5717144641037335,
        100.0: 0.5197708816462203,
        1000.0: 0.5150882610908489
    }
    for alpha, entropy in alpha_to_entropies.items():
        assert pyqentangle.renyi_entanglement_entropy(schmidt_modes, alpha) / shannon == pytest.approx(entropy)
