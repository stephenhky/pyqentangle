
import unittest2

import numpy as np
from scipy.integrate import dblquad

from pyqentangle.quantumstates.harmonics import disentangled_gaussian_wavefcn, coupled_excited_harmonics, correlated_bipartite_gaussian_wavefcn


normsq = lambda x: x*np.conj(x)


class testHarmonicsNorm(unittest2.TestCase):
    def test_disentangled_gaussian(self):
        norm, err = dblquad(lambda x1, x2: normsq(disentangled_gaussian_wavefcn()(x1, x2)),
                            -np.inf, np.inf, lambda x: -np.inf, lambda y: np.inf)
        self.assertAlmostEqual(norm, 1, delta=abs(err))

    def test_correlated_bipartite_gaussian(self):
        covmatrix = np.array([[2., 0.5], [0.5, 1.]])
        norm, err = dblquad(lambda x1, x2: normsq(correlated_bipartite_gaussian_wavefcn(covmatrix)(x1, x2)),
                            -np.inf, np.inf, lambda x: -np.inf, lambda y: np.inf)
        self.assertAlmostEqual(norm, 1, delta=abs(err))

    def test_first_excited_state(self):
        for n in range(20):
            norm, err = dblquad(lambda x1, x2: normsq(coupled_excited_harmonics(n)(x1, x2)),
                                -np.inf, np.inf, lambda x: -np.inf, lambda y: np.inf)
            self.assertAlmostEqual(norm, 1, delta=abs(err))
