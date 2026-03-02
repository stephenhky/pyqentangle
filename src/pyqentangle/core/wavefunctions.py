
from abc import ABC, abstractmethod
from types import LambdaType, FunctionType
from typing import Union, Self

import numpy as np
import numpy.typing as npt


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
            def __call__(self, coordinates):
                return self.__call__(coordinates) + other.__call__(coordinates)
        return ResultingAddedWavefunction()

    def __mul__(self, other: Self) -> Self:
        class ResultingMulWaveFunction(WaveFunction):
            def __call__(self, coordiniates):
                return self.__call__(coordiniates) * other.__call__(coordiniates)
        return ResultingMulWaveFunction()

    def __rmul__(self, other: Union[float, np.complex128]) -> Self:
        class ResultingScalarMulWaveFunction(WaveFunction):
            def __call__(self, coordiniates):
                return self.__call__(coordiniates) * other
        return ResultingScalarMulWaveFunction()



class AnalyticWaveFunction(ABC, WaveFunction):
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
