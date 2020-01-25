
import unittest2

import numpy as np

from pyqentangle.tncompute import bipartitepurestate_densitymatrix, \
    bipartitepurestate_partialtranspose_densitymatrix, flatten_bipartite_densitymatrix


class testContinuousEntanglement(unittest2.TestCase):
    def setUp(self):
        self.tensor = np.array([[0., np.sqrt(0.6) * 1j], [np.sqrt(0.4) * 1j, 0.]])

    def tearDown(self):
        pass

    def test_partialtranpose(self):
        fullden = bipartitepurestate_densitymatrix(self.tensor)
        fullden_pt0 = bipartitepurestate_partialtranspose_densitymatrix(self.tensor, 0)
        fullden_pt1 = bipartitepurestate_partialtranspose_densitymatrix(self.tensor, 1)

        self.assertAlmostEqual(fullden[0, 1, 0, 1], 0.6+0j)
        self.assertAlmostEqual(fullden[0, 1, 1, 0], np.sqrt(0.6*0.4)+0j)
        self.assertAlmostEqual(fullden[1, 0, 0, 1], np.sqrt(0.6*0.4)+0j)
        self.assertAlmostEqual(fullden[1, 0, 1, 0], 0.4+0j)

        self.assertAlmostEqual(fullden_pt0[0, 1, 0, 1], 0.6+0j)
        self.assertAlmostEqual(fullden_pt0[1, 1, 0, 0], np.sqrt(0.6*0.4)+0j)
        self.assertAlmostEqual(fullden_pt0[0, 0, 1, 1], np.sqrt(0.6*0.4)+0j)
        self.assertAlmostEqual(fullden_pt0[1, 0, 1, 0], 0.4+0j)

        self.assertAlmostEqual(fullden_pt1[0, 1, 0, 1], 0.6+0j)
        self.assertAlmostEqual(fullden_pt1[0, 0, 1, 1], np.sqrt(0.6*0.4)+0j)
        self.assertAlmostEqual(fullden_pt1[1, 1, 0, 0], np.sqrt(0.6*0.4)+0j)
        self.assertAlmostEqual(fullden_pt1[1, 0, 1, 0], 0.4+0j)


    def test_flatten(self):
        fullden = bipartitepurestate_densitymatrix(self.tensor)
        fullden_pt0 = bipartitepurestate_partialtranspose_densitymatrix(self.tensor, 0)
        fullden_pt1 = bipartitepurestate_partialtranspose_densitymatrix(self.tensor, 1)

        flatten_fullden = flatten_bipartite_densitymatrix(fullden)
        flatten_fullden_pt0 = flatten_bipartite_densitymatrix(fullden_pt0)
        flatten_fullden_pt1 = flatten_bipartite_densitymatrix(fullden_pt1)

        self.assertAlmostEqual(flatten_fullden[1, 1], 0.6+0j)
        self.assertAlmostEqual(flatten_fullden[1, 2], np.sqrt(0.6*0.4)+0j)
        self.assertAlmostEqual(flatten_fullden[2, 1], np.sqrt(0.6*0.4)+0j)
        self.assertAlmostEqual(flatten_fullden[2, 2], 0.4+0j)

        self.assertAlmostEqual(flatten_fullden_pt0[1, 1], 0.6+0j)
        self.assertAlmostEqual(flatten_fullden_pt0[3, 0], np.sqrt(0.6*0.4)+0j)
        self.assertAlmostEqual(flatten_fullden_pt0[0, 3], np.sqrt(0.6*0.4)+0j)
        self.assertAlmostEqual(flatten_fullden_pt0[2, 2], 0.4+0j)

        self.assertAlmostEqual(flatten_fullden_pt1[1, 1], 0.6+0j)
        self.assertAlmostEqual(flatten_fullden_pt1[0, 3], np.sqrt(0.6*0.4)+0j)
        self.assertAlmostEqual(flatten_fullden_pt1[3, 0], np.sqrt(0.6*0.4)+0j)
        self.assertAlmostEqual(flatten_fullden_pt1[2, 2], 0.4+0j)



if __name__ == '__main__':
    unittest2.main()