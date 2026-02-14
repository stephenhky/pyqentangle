
import numpy as np
import pytest

from pyqentangle.continuous import numerical_continuous_function


def test_interpolation_real():
    xarray = np.array([0., 1., 2., 3.])
    yarray = np.array([1., 3., 5., 7.])
    f = numerical_continuous_function(xarray, yarray)
    ys = f(np.array([1.5, 2.5]))
    assert ys[0] == pytest.approx(4.)
    assert ys[1] ==pytest.approx(6.)


def test_interpolation_complex():
    xarray = np.array([0., 1., 2., 3.])
    yarray = np.array([0., 1.+1.j, 2.+2.j, 3.+3.j])
    f = numerical_continuous_function(xarray, yarray)
    ys = f(np.array([1.5, 2.5]))
    assert ys[0] == pytest.approx(1.5+1.5j)
    assert ys[1] == pytest.approx(2.5+2.5j)
