
from abc import ABC, abstractmethod
from types import LambdaType, FunctionType
from typing import Union

import numpy as np
import numpy.typing as npt


class WaveFunction(ABC):
    @abstractmethod
    def __call__(
            self,
            coordinates: Union[npt.NDArray[np.float64], float]
    ) -> npt.NDArray[np.complex128]:
        raise NotImplemented()


class AnalyticWaveFunction(ABC, WaveFunction):
    def __init__(self, lambda_func: Union[LambdaType, FunctionType]):
        self._lambda_func = np.vectorize(lambda_func)


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
