
import unittest2

import numpy as np

import pyqentangle

class testDiscreteEntanglement(unittest2.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testSchmidt(self):
        tensor = np.array([[0., np.sqrt(0.6)*1j], [np.sqrt(0.4)*1j, 0.]])
        for approach in ['tensornetwork', 'numpy']:
            modes = pyqentangle.schmidt_decomposition(tensor, approach=approach)
            self.assertAlmostEqual(modes[0][0], np.sqrt(0.6))
            self.assertAlmostEqual(pyqentangle.entanglement_entropy(modes), 0.6730116670092563)
            self.assertAlmostEqual(pyqentangle.participation_ratio(modes), 1.9230769230769227)
            self.assertAlmostEqual(pyqentangle.negativity(tensor), 0.489897948556636)


if __name__ == '__main__':
    unittest2.main()