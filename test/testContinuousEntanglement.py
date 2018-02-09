
import unittest2

import numpy as np

import pyqentangle

class testContinuousEntanglement(unittest2.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testEntangledOscillators(self):
        fcn = lambda x1, x2: np.exp(-((0.5 * (x1 + x2)) ** 2)) * np.exp(-(x1 - x2) ** 2) * np.sqrt(2. / np.pi)
        decompositions = pyqentangle.continuous_schmidt_decomposition(fcn, -10., 10., -10., 10., keep=10)
        eigenvalues = map(lambda item: item[0], decompositions)
        self.assertAlmostEqual(eigenvalues[0], 0.888888889)
        self.assertAlmostEqual(eigenvalues[1], 0.098765432)
        # self.assertAlmostEqual(quad(lambda x1: decompositions[0][1](x1)*decompositions[0][1](x1), -10, 10), 1)
        # self.assertAlmostEqual(quad(lambda x2: decompositions[0][2](x2)*decompositions[0][2](x2), -10, 10), 1)

if __name__ == '__main__':
    unittest2.main()