
from abc import ABC, abstractmethod
from types import LambdaType, FunctionType
from typing import Union
import sys

import numpy as np
import numpy.typing as npt

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from .interpolate import numerical_continuous_interpolation


class WaveFunction(ABC):
    """Abstract base class representing a quantum wavefunction.

    Subclasses must implement :meth:`__call__` to evaluate the wavefunction
    at given coordinates. Provides arithmetic operations (addition and
    scalar/wavefunction multiplication) and a probability density helper.
    """

    @abstractmethod
    def __call__(
            self,
            coordinates: Union[npt.NDArray[np.float64], float]
    ) -> npt.NDArray[np.complex128]:
        """Evaluate the wavefunction at the given coordinates.

        Args:
            coordinates (numpy.ndarray or float): Coordinates at which to evaluate the wavefunction.

        Returns:
            numpy.ndarray: Complex amplitude(s) at the given coordinates.
        """
        raise NotImplemented()

    def prob_density(self, coordinates: Union[npt.NDArray[np.float64], float]) -> float:
        """Compute the probability density at the given coordinates.

        The probability density is defined as :math:`|\\psi(x)|^2 = \\text{Re}(\\psi)^2 + \\text{Im}(\\psi)^2`.

        Args:
            coordinates (numpy.ndarray or float): Coordinates at which to evaluate the probability density.

        Returns:
            float: Probability density at the given coordinates.
        """
        amplitude = self.__call__(coordinates)
        return np.square(np.real(amplitude)) + np.square(np.imag(amplitude))

    def __add__(self, other: Self) -> Self:
        """Return a new wavefunction that is the sum of this and another wavefunction.

        Args:
            other (WaveFunction): The wavefunction to add.

        Returns:
            WaveFunction: A new wavefunction representing the pointwise sum.
        """
        class ResultingAddedWavefunction(WaveFunction):
            def __call__(self2, coordinates):
                return self.__call__(coordinates) + other.__call__(coordinates)
        return ResultingAddedWavefunction()

    def __mul__(self, other: Union[Self, float, np.complex128]) -> Self:
        """Return a new wavefunction that is the product of this wavefunction with another wavefunction or scalar.

        Args:
            other (WaveFunction, float, or complex): The wavefunction or scalar to multiply by.

        Returns:
            WaveFunction: A new wavefunction representing the pointwise product or scalar multiplication.
        """
        if isinstance(other, WaveFunction):
            class ResultingMulWaveFunction(WaveFunction):
                def __call__(self2, coordiniates):
                    return self.__call__(coordiniates) * other.__call__(coordiniates)
            return ResultingMulWaveFunction()
        else:
            class ResultingScalarMulWaveFunction(WaveFunction):
                def __call__(self2, coordiniates):
                    return self.__call__(coordiniates) * other
            return ResultingScalarMulWaveFunction()

    def __rmul__(self, other: Union[Self, float, np.complex128]) -> Self:
        """Return a new wavefunction that is the product of a scalar or wavefunction with this wavefunction.

        This supports left-multiplication by a scalar or another wavefunction.

        Args:
            other (WaveFunction, float, or complex): The scalar or wavefunction to multiply by.

        Returns:
            WaveFunction: A new wavefunction representing the product.
        """
        return self.__mul__(other)


class AnalyticWaveFunction(WaveFunction, ABC):
    """Abstract base class for analytically defined wavefunctions.

    Subclasses represent wavefunctions that are defined by an analytic (closed-form)
    expression rather than by numerical interpolation.
    """
    pass


class Analytic1DWaveFunction(AnalyticWaveFunction):
    """A one-dimensional analytically defined wavefunction.

    Wraps a callable (e.g., a lambda or regular function) that maps a scalar
    or 1-D array of coordinates to complex amplitudes. The callable is optionally
    vectorized using :func:`numpy.vectorize`.
    """

    def __init__(
            self,
            lambda_func: Union[LambdaType, FunctionType],
            to_vectorize: bool=True
    ):
        """Initialize the 1D analytic wavefunction.

        Args:
            lambda_func (callable): A function of one variable returning the wavefunction amplitude.
            to_vectorize (bool, optional): Whether to wrap `lambda_func` with :func:`numpy.vectorize`.
                Defaults to ``True``.
        """
        if to_vectorize:
            self._lambda_func = np.vectorize(lambda_func)
        else:
            self._lambda_func = lambda_func

    def __call__(
            self,
            coordinates: Union[npt.NDArray[np.float64], float]
    ) -> npt.NDArray[np.complex128]:
        """Evaluate the wavefunction at the given coordinates.

        Args:
            coordinates (numpy.ndarray or float): Coordinate(s) at which to evaluate the wavefunction.
                A float is automatically converted to a 1-element array.

        Returns:
            numpy.ndarray: Complex amplitude(s) at the given coordinates.
        """
        if isinstance(coordinates, float):
            coordinates = np.array([coordinates])
        return self._lambda_func(coordinates)


class AnalyticMultiDimWaveFunction(AnalyticWaveFunction):
    """A multi-dimensional analytically defined wavefunction.

    Wraps a callable that accepts a coordinate array (1-D for a single point,
    or 2-D for a batch of points) and returns the complex amplitude. The callable
    must **not** be a :class:`numpy.vectorize` instance, as vectorization is
    handled internally.
    """

    def __init__(
            self,
            lambda_func: Union[LambdaType, FunctionType]   # do not put a vectorize function
    ):
        """Initialize the multi-dimensional analytic wavefunction.

        Args:
            lambda_func (callable): A function accepting a coordinate array and returning the
                wavefunction amplitude. Must not be a :class:`numpy.vectorize` instance.

        Raises:
            ValueError: If `lambda_func` is a :class:`numpy.vectorize` instance.
        """
        if isinstance(lambda_func, np.vectorize):
            raise ValueError("Do not pass a numpy.vectorize function.")
        self._lambda_func = lambda_func

    def __call__(
            self,
            coordinates: Union[npt.NDArray[np.float64], float]
    ) -> npt.NDArray[np.complex128]:
        """Evaluate the wavefunction at the given coordinates.

        Args:
            coordinates (numpy.ndarray): Coordinate array. Must be 1-D (single point) or
                2-D (batch of points, shape ``(N, D)``). Floats and integers are not accepted.

        Returns:
            numpy.ndarray: Complex amplitude(s) at the given coordinates.

        Raises:
            TypeError: If `coordinates` is a float or int.
            ValueError: If `coordinates` has an unsupported number of dimensions.
        """
        if isinstance(coordinates, float) or isinstance(coordinates, int):
            raise TypeError("It has to be a coordinates in an array form, not a float number.")

        if coordinates.ndim == 1:
            return self._lambda_func(coordinates)
        elif coordinates.ndim == 2:
            return np.array([
                self._lambda_func(coordinates[i, :])
                for i in range(coordinates.shape[0])
            ])
        else:
            raise ValueError(f"The coordinates have the wrong shape: {coordinates.shape}")


class InterpolatingWaveFunction(WaveFunction):
    """A wavefunction defined by numerical interpolation over a discrete grid.

    Given arrays of independent variable values (`xarray`) and corresponding
    complex amplitudes (`yarray`), this wavefunction evaluates at arbitrary
    coordinates using linear interpolation via :func:`~pyqentangle.core.interpolate.numerical_continuous_interpolation`.
    """

    def __init__(
            self,
            xarray: npt.NDArray[np.float64],
            yarray: npt.NDArray[np.complex128]
    ):
        """Initialize the interpolating wavefunction.

        Args:
            xarray (numpy.ndarray): Array of independent variable values (grid points).
            yarray (numpy.ndarray): Array of complex wavefunction amplitudes at the grid points.
        """
        self._xarray = xarray
        self._yarray = yarray

    def __call__(
            self,
            coordinates: Union[npt.NDArray[np.float64], float]
    ) -> npt.NDArray[np.complex128]:
        """Evaluate the wavefunction at the given coordinates using interpolation.

        Args:
            coordinates (numpy.ndarray or float): Coordinate(s) at which to evaluate the wavefunction.
                A float evaluates at a single point; an array evaluates at each element.

        Returns:
            numpy.ndarray: Interpolated complex amplitude(s) at the given coordinates.
        """
        if isinstance(coordinates, float):
            return numerical_continuous_interpolation(self._xarray, self._yarray, coordinates)
        else:
            return np.array([
                numerical_continuous_interpolation(self._xarray, self._yarray, coordinates[i])
                for i in range(coordinates.shape[0])
            ])
