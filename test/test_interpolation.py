
import unittest2

import numpy as np

from pyqentangle.continuous import numerical_continuous_function


class TestInterpolation(unittest2.TestCase):
    def test_real(self):
        xarray = np.array([0., 1., 2., 3.])
        yarray = np.array([1., 3., 5., 7.])
        f = numerical_continuous_function(xarray, yarray)
        y1 = f(1.5)
        self.assertAlmostEqual(y1, 4.)
        y2 = f(2.5)
        self.assertAlmostEqual(y2, 6.)

    def test_complex(self):
        xarray = np.array([0., 1., 2., 3.])
        yarray = np.array([0., 1.+1.j, 2.+2.j, 3.+3.j])
        f = numerical_continuous_function(xarray, yarray)
        y1 = f(1.5)
        self.assertAlmostEqual(y1, 1.5+1.5j)
        y2 = f(2.5)
        self.assertAlmostEqual(y2, 2.5+2.5j)