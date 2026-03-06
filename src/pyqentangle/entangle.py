
from typing import Union, Literal, Generator, Optional

import numpy as np
import numpy.typing as npt

from .core.schmidt import schmidt_decomposition
from .core.continuous import continuous_schmidt_decomposition
from .core.wavefunctions import WaveFunction, AnalyticMultiDimWaveFunction
from .schemas import DiscreteSchmidtMode, ContinuousSchmidtMode


class DiscreteSchmidtDecomposer:
    def __init__(
            self,
            tensor: Union[npt.NDArray[np.complex128], npt.NDArray[np.float64]],
            lazy: bool = False,
            approach: Literal["tensornetwork", "numpy"] = "tensornetwork"
    ):
        self._tensor = tensor
        self._lazy = lazy
        self._approach = approach

        self._calculated = False
        if not self._lazy:
            self._compute()

    def _compute(self) -> None:
        raw_decomposed_results = schmidt_decomposition(self._tensor, self._approach)
        self._results = [
            DiscreteSchmidtMode(schmidt_coef=item[0], mode1=item[1], mode2=item[2])
            for item in raw_decomposed_results
        ]
        self._calculated = True

    def modes(self) -> list[DiscreteSchmidtMode]:
        if not self._calculated:
            # lazy mode, not calculated previously
            self._compute()
        return self._results

    def mode_iterator(self) -> Generator[DiscreteSchmidtMode, None, None]:
        if not self._calculated:
            # lazy mode, not calculated previously
            self._compute()
        for result in self._results:
            yield result

    @property
    def tensor(self) -> Union[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
        return self._tensor

    @property
    def calculated(self) -> bool:
        return self._calculated

    @property
    def approach(self) -> Literal["tensornetwork", "numpy"]:
        return self._approach


class ContinuousSchmidtDecomposer:
    def __init__(
            self,
            bipartite_wavefunction: callable,
            x1_lo: float,
            x1_hi: float,
            x2_lo: float,
            x2_hi: float,
            nb_x1: int = 100,
            nb_x2: int = 100,
            keep: Optional[int] = None,
            lazy: bool = False,
            approach: Literal["tensornetwork", "numpy"] = 'tensornetwork'
    ):
        if not isinstance(bipartite_wavefunction, WaveFunction):
            self._bipartitle_wavefunction = AnalyticMultiDimWaveFunction(bipartite_wavefunction)
        else:
            self._bipartitle_wavefunction = bipartite_wavefunction
        self._x1_lo = x1_lo
        self._x1_hi = x1_hi
        self._x2_lo = x2_lo
        self._x2_hi = x2_hi
        self._nb_x1 = nb_x1
        self._nb_x2 = nb_x2
        self._keep = keep
        self._lazy = lazy
        self._approach = approach

        self._calculated = False
        if not self._lazy:
            self._compute()

    def _compute(self) -> None:
        raw_decomposition_results = continuous_schmidt_decomposition(
            self._bipartitle_wavefunction,
            self._x1_lo,
            self._x1_hi,
            self._x2_lo,
            self._x2_hi,
            nb_x1=self._nb_x1,
            nb_x2=self._nb_x2,
            keep=self._keep,
            approach=self._approach
        )
        self._results = [
            ContinuousSchmidtMode(
                schmidt_coef=item[0],
                wavefunction1=item[1],
                wavefunction2=item[2]
            )
            for item in raw_decomposition_results
        ]

    def modes(self) -> list[ContinuousSchmidtMode]:
        if not self._calculated:
            # lazy mode, not calculated previously
            self._compute()
        return self._results

    def mode_iterator(self) -> Generator[ContinuousSchmidtMode, None, None]:
        if not self._calculated:
            # lazy mode, not calculated previously
            self._compute()
        for result in self._results:
            yield result

    @property
    def bipartite_wavefuncion(self) -> WaveFunction:
        return self._bipartitle_wavefunction

    @property
    def x1_lo(self) -> float:
        return self._x1_lo

    @property
    def x1_hi(self) -> float:
        return self._x1_hi

    @property
    def x2_lo(self) -> float:
        return self._x2_lo

    @property
    def x2_hi(self) -> float:
        return self._x2_hi

    @property
    def nb_x1(self) -> int:
        return self._nb_x1

    @property
    def nb_x2(self) -> int:
        return self._nb_x2

    @property
    def calculated(self) -> bool:
        return self._calculated

    @property
    def approach(self) -> Literal["tensornetwork", "numpy"]:
        return self._approach

