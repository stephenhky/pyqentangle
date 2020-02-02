import unittest2

import numpy as np

import pyqentangle
import pyqentangle.tncompute


class testRedDenMat(unittest2.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testTwoLevel(self):
        tensor = np.array([[0., np.sqrt(0.6) * 1j], [np.sqrt(0.4) * 1j, 0.]])

        # subsystem A
        reddenmat0 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 0)
        self.assertAlmostEqual(np.trace(reddenmat0), 1.0)
        self.assertAlmostEqual(reddenmat0[0, 0], 0.6)
        self.assertAlmostEqual(reddenmat0[1, 1], 0.4)
        self.assertAlmostEqual(reddenmat0[0, 1], 0.)
        self.assertAlmostEqual(reddenmat0[1, 0], 0.)

        # subsystem B
        reddenmat1 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 1)
        self.assertAlmostEqual(np.trace(reddenmat1), 1.0)
        self.assertAlmostEqual(reddenmat1[0, 0], 0.4)
        self.assertAlmostEqual(reddenmat1[1, 1], 0.6)
        self.assertAlmostEqual(reddenmat1[0, 1], 0.)
        self.assertAlmostEqual(reddenmat1[1, 0], 0.)

    def testThreeLevel(self):
        tensor = np.array([[np.sqrt(0.5), 0.0, 0.0], [0.0, np.sqrt(0.3)*1.j, 0.0], [0.0, 0.0, np.sqrt(0.2)]])

        # subsystem A
        reddenmat0 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 0)
        self.assertAlmostEqual(np.trace(reddenmat0), 1.0)
        self.assertAlmostEqual(reddenmat0[0, 0], 0.5)
        self.assertAlmostEqual(reddenmat0[1, 1], 0.3)
        self.assertAlmostEqual(reddenmat0[2, 2], 0.2)
        self.assertAlmostEqual(reddenmat0[1, 0], 0.)
        self.assertAlmostEqual(reddenmat0[1, 2], 0.)
        self.assertAlmostEqual(reddenmat0[2, 0], 0.)

        # subsystem B
        reddenmat1 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 1)
        self.assertAlmostEqual(np.trace(reddenmat1), 1.0)
        self.assertAlmostEqual(reddenmat1[0, 0], 0.5)
        self.assertAlmostEqual(reddenmat1[1, 1], 0.3)
        self.assertAlmostEqual(reddenmat1[2, 2], 0.2)
        self.assertAlmostEqual(reddenmat1[2, 1], 0.)
        self.assertAlmostEqual(reddenmat1[1, 0], 0.)
        self.assertAlmostEqual(reddenmat1[0, 2], 0.)

    def testTwoLevels2(self):
        tensor = np.array([[np.sqrt(0.5), np.sqrt(0.25) * 1.j], [0., np.sqrt(0.25)]])

        # subsystem A
        reddenmat0 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 0)
        self.assertAlmostEqual(np.trace(reddenmat0), 1.0)
        self.assertAlmostEqual(reddenmat0[0, 0], 0.75)
        self.assertAlmostEqual(reddenmat0[1, 1], 0.25)
        self.assertAlmostEqual(reddenmat0[0, 1], 0.+0.25j)
        self.assertAlmostEqual(reddenmat0[1, 0], 0.-0.25j)

        # subsystem B
        reddenmat1 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 1)
        self.assertAlmostEqual(np.trace(reddenmat1), 1.0)
        self.assertAlmostEqual(reddenmat1[0, 0], 0.5)
        self.assertAlmostEqual(reddenmat1[1, 1], 0.5)
        self.assertAlmostEqual(reddenmat1[0, 1], 0.-0.35355339j)
        self.assertAlmostEqual(reddenmat1[1, 0], 0.+0.35355339j)


    def testFiftennLevels(self):
        tensor = np.zeros((15, 15))
        tensor[1, 1] = np.sqrt(0.3)
        tensor[1, 2] = np.sqrt(0.1)
        tensor[1, 3] = np.sqrt(0.1)
        tensor[11, 10] = np.sqrt(0.2)
        tensor[11, 11] = np.sqrt(0.1)
        tensor[11, 12] = np.sqrt(0.2)
        reddenmat0 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 0)
        reddenmat1 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 0)
        self.assertAlmostEqual(np.trace(reddenmat0), 1.0)
        self.assertAlmostEqual(np.trace(reddenmat1), 1.0)


if __name__ == '__main__':
    unittest2.main()