
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from ..core.wavefunctions import WaveFunction


@dataclass
class DiscreteSchmidtMode:
    schmidt_coef: Optional[float]
    mode1: npt.NDArray[np.complex128]
    mode2: npt.NDArray[np.complex128]


@dataclass
class ContinuousSchmidtMode:
    schmidt_coef: Optional[float]
    wavefunction1: WaveFunction
    wavefunction2: WaveFunction
