
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
    @abstractmethod
    def __call__(
            self,
            coordinates: Union[npt.NDArray[np.float64], float]
    ) -> npt.NDArray[np.complex128]:
        raise NotImplemented()

    def prob_density(self, coordinates: Union[npt.NDArray[np.float64], float]) -> float:
        amplitude = self.__call__(coordinates)
        return np.square(np.real(amplitude)) + np.square(np.imag(amplitude))

    def __add__(self, other: Self) -> Self:
        class ResultingAddedWavefunction(WaveFunction):
            def __call__(self2, coordinates):
                return self.__call__(coordinates) + other.__call__(coordinates)
        return ResultingAddedWavefunction()

    def __mul__(self, other: Union[Self, float, np.complex128]) -> Self:
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
        return self.__mul__(other)


class AnalyticWaveFunction(WaveFunction, ABC):
    def __init__(
            self,
            lambda_func: Union[LambdaType, FunctionType],
            to_vectorize: bool=True
    ):
        if to_vectorize:
            self._lambda_func = np.vectorize(lambda_func)
        else:
            self._lambda_func = lambda_func


class Analytic1DWaveFunction(AnalyticWaveFunction):
    def __call__(
            self,
            coordinates: Union[npt.NDArray[np.float64], float]
    ) -> npt.NDArray[np.complex128]:
        if isinstance(coordinates, float):
            coordinates = np.array([coordinates])
        return self._lambda_func(coordinates)


class AnalyticMultiDimWaveFunction(AnalyticWaveFunction):
    def __call__(
            self,
            coordinates: Union[npt.NDArray[np.float64], float]
    ) -> npt.NDArray[np.complex128]:
        if isinstance(coordinates, float):
            raise TypeError("It has to be a coordinates in an array form, not a float number.")

        if coordinates.ndim == 1:
            coordinates = np.array([coordinates])

        return self._lambda_func(coordinates)


class InterpolatingWaveFunction(WaveFunction):
    def __init__(
            self,
            xarray: npt.NDArray[np.float64],
            yarray: npt.NDArray[np.complex128]
    ):
        self._xarray = xarray
        self._yarray = yarray

    def __call__(
            self,
            coordinates: Union[npt.NDArray[np.float64], float]
    ) -> npt.NDArray[np.complex128]:
        if isinstance(coordinates, float):
            return numerical_continuous_interpolation(self._xarray, self._yarray, coordinates)
        elif coordinates.ndim == 1:
            return numerical_continuous_interpolation(self._xarray, self._yarray, coordinates)
        else:
            return np.array([
                numerical_continuous_interpolation(self._xarray, self._yarray, coordinates[i])
                for i in range(coordinates.shape[0])
            ])
