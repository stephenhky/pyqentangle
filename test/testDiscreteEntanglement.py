
import unittest2

import numpy as np

import pyqentangle

class testDiscreteEntanglement(unittest2.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testSchmidt(self):
        tensor = np.array([[0., np.sqrt(0.5)], [np.sqrt(0.5), 0.]])
        modes = pyqentangle.schmidt_decomposition(tensor)
        self.assertAlmostEqual(modes[0][0], 0.5)

if __name__ == '__main__':
    unittest2.main()