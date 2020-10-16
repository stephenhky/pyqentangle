
import unittest2

import numpy as np

import pyqentangle


class TestContinuousComplex(unittest2.TestCase):
    def test_imagpart(self):
        f1 = lambda x1, x2: np.exp(-0.5 * (x1 + x2) ** 2) * np.exp(-(x1 - x2) ** 2) * np.sqrt(np.sqrt(8.) / np.pi)
        f2 = lambda x1, x2: f1(x1, x2) * np.sqrt(0.5) * (1 + 1j)

        f1_modes = pyqentangle.continuous_schmidt_decomposition(f1, -10, 10, -10, 10, keep=5)
        f2_modes = pyqentangle.continuous_schmidt_decomposition(f2, -10, 10, -10, 10, keep=5)

        for f1_mode, f2_mode in zip(f1_modes, f2_modes):
            f1_eigval = f1_mode[0]
            f2_eigval = f2_mode[0]
            np.testing.assert_almost_equal(f1_eigval, f2_eigval)


if __name__ == '__main__':
    unittest2.main()
