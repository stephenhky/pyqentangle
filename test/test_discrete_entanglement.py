
import numpy as np
import pytest

import pyqentangle


def test_schmidt_decomposition():
    tensor = np.array([[0., np.sqrt(0.6)*1j], [np.sqrt(0.4)*1j, 0.]])
    for approach in ['tensornetwork', 'numpy']:
        modes = pyqentangle.schmidt_decomposition(tensor, approach=approach)
        assert modes[0][0] == pytest.approx(np.sqrt(0.6))
        assert pyqentangle.entanglement_entropy(modes) == pytest.approx(0.6730116670092563)
        assert pyqentangle.participation_ratio(modes) == pytest.approx(1.9230769230769227)
        assert pyqentangle.negativity(tensor) == pytest.approx(0.489897948556636)
