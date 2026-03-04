
import numpy as np
import pytest

from pyqentangle.core.interpolate import numerical_continuous_function, numerical_continuous_interpolation
from pyqentangle.core.wavefunctions import InterpolatingWaveFunction


def test_interpolation():
    xarray = np.array([0., 1., 2.])
    yarray = np.array([0., 1., 4.], dtype=np.complex128)
    assert np.real(numerical_continuous_interpolation(xarray, yarray, 0.5)) == pytest.approx(0.5)
    assert np.real(numerical_continuous_interpolation(xarray, yarray, 1.5)) == pytest.approx(2.5)


def test_interpolation_real_deprecated():
    xarray = np.array([0., 1., 2., 3.])
    yarray = np.array([1., 3., 5., 7.], dtype=np.complex128)
    f = numerical_continuous_function(xarray, yarray)
    ys = f(np.array([1.5, 2.5]))
    assert ys[0] == pytest.approx(4.)
    assert ys[1] == pytest.approx(6.)

def test_interpolation_real():
    xarray = np.array([0., 1., 2., 3.])
    yarray = np.array([1., 3., 5., 7.], dtype=np.complex128)
    wavefunction = InterpolatingWaveFunction(xarray, yarray)
    assert wavefunction(1.5) == pytest.approx(4.)
    assert wavefunction(2.5) == pytest.approx(6.)
    np.testing.assert_array_almost_equal(wavefunction(np.array([1.5, 2.5])), np.array([4., 6.]))


def test_interpolation_complex_deprecated():
    xarray = np.array([0., 1., 2., 3.])
    yarray = np.array([0., 1.+1.j, 2.+2.j, 3.+3.j])
    f = numerical_continuous_function(xarray, yarray)
    ys = f(np.array([1.5, 2.5]))
    assert ys[0] == pytest.approx(1.5+1.5j)
    assert ys[1] == pytest.approx(2.5+2.5j)

def test_interpolation_complex():
    xarray = np.array([0., 1., 2., 3.])
    yarray = np.array([0., 1. + 1.j, 2. + 2.j, 3. + 3.j])
    wavefunction = InterpolatingWaveFunction(xarray, yarray)
    assert wavefunction(1.5) == pytest.approx(1.5+1.5j)
    assert wavefunction(2.5) == pytest.approx(2.5+2.5j)
    np.testing.assert_array_almost_equal(wavefunction(np.array([1.5, 2.5])), np.array([1.5+1.5j, 2.5+2.5j]))
