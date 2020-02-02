
import unittest2

import numpy as np

import pyqentangle
import pyqentangle.tncompute


class TestFullDen(unittest2.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_uneventwolevels(self):
        tensor = np.array([[0., np.sqrt(np.reciprocal(3.))],
                           [-np.sqrt(2./3.)*1j, 0.]])
        fulldenmat = pyqentangle.tncompute.bipartitepurestate_densitymatrix(tensor)

        self.assertAlmostEqual(fulldenmat[0, 1, 0, 1], np.reciprocal(3.))
        self.assertAlmostEqual(fulldenmat[1, 0, 1, 0], 2./3.)
        self.assertAlmostEqual(fulldenmat[0, 1, 1, 0], np.sqrt(2)/3*1j)
        self.assertAlmostEqual(fulldenmat[1, 0, 0, 1], -np.sqrt(2) / 3 * 1j)

if __name__ == '__main__':
    unittest2.main()
