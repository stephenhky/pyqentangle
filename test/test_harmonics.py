
import unittest2

import numpy as np
from scipy.integrate import dblquad

from pyqentangle.quantumstates.harmonics import disentangled_gaussian_wavefcn, coupled_excited_harmonics


class testHarmonicsNorm(unittest2.TestCase):
    def disentangled_gaussian(self):
        norm, err = dblquad(lambda x1, x2: disentangled_gaussian_wavefcn(x1, x2),
                            -np.inf, np.inf, lambda x: -np.inf, lambda y: np.inf)
        self.assertAlmostEqual(norm, 1, delta=abs(err))

    def first_excited_state(self):
        for n in range(20):
            norm, err = dblquad(lambda x1, x2: coupled_excited_harmonics(n)(x1, x2),
                                -np.inf, np.inf, lambda x: -np.inf, lambda y: np.inf)
            self.assertAlmostEqual(norm, 1, delta=abs(err))