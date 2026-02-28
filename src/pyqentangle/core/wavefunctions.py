
from abc import ABC, abstractmethod
from types import LambdaType

import numpy as np
import numpy.typing as npt


class WaveFunction(ABC):
    @abstractmethod
    def __call__(
            self,
            coordinates: npt.NDArray[np.complex128] | npt.NDArray[np.float64] | float
    ) -> npt.NDArray[np.complex128]:
        raise NotImplemented()


class AnalyticWaveFunction(WaveFunction):
    def __init__(self, lambda_func: LambdaType):
        self._lambda_func = np.vectorize(lambda_func)

    def __call__(
            self,
            coordinates: npt.NDArray[np.complex128] | npt.NDArray[np.float64] | float
    ) -> npt.NDArray[np.complex128]:
        if isinstance(coordinates, float):
            coordinates = np.array(coordinates)
        return self._lambda_func(coordinates)
