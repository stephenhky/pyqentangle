
from dataclasses import dataclass
from abc import ABC

import numpy as np
import numpy.typing as npt

from ..core.wavefunctions import WaveFunction


@dataclass
class SchmidtMode(ABC):
    schmidt_coef: float


@dataclass
class DiscreteSchmidtMode(SchmidtMode):
    mode1: npt.NDArray[np.complex128]
    mode2: npt.NDArray[np.complex128]


@dataclass
class ContinuousSchmidtMode(SchmidtMode):
    wavefunction1: WaveFunction
    wavefunction2: WaveFunction
