
import unittest2

import numpy as np
import pyqentangle

class testConcurrence(unittest2.TestCase):
    def testSinglet(self):
        singlet = np.array([[0., 1.], [1.j, 0.]]) / np.sqrt(2)
        self.assertAlmostEqual(pyqentangle.concurrence(singlet), 1., delta=1e-6)


if __name__ == '__main__':
    unittest2.main()