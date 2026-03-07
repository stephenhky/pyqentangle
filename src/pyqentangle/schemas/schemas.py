
from dataclasses import dataclass
from abc import ABC

import numpy as np
import numpy.typing as npt

from ..core.wavefunctions import WaveFunction


@dataclass
class SchmidtMode(ABC):
    """Abstract base dataclass representing a single Schmidt mode.

    A Schmidt mode pairs a Schmidt coefficient with the corresponding
    eigenmodes of the two subsystems. Concrete subclasses provide the
    eigenmode representation appropriate for discrete or continuous systems.

    Attributes:
        schmidt_coef (float): The Schmidt coefficient (singular value) for this mode.
    """

    schmidt_coef: float


@dataclass
class DiscreteSchmidtMode(SchmidtMode):
    """A Schmidt mode for a discrete bipartite quantum system.

    Extends :class:`SchmidtMode` with the eigenvectors of both subsystems
    as complex-valued NumPy arrays.

    Attributes:
        schmidt_coef (float): The Schmidt coefficient (singular value) for this mode.
        mode1 (numpy.ndarray): Eigenvector of the first subsystem.
        mode2 (numpy.ndarray): Eigenvector of the second subsystem.
    """

    mode1: npt.NDArray[np.complex128]
    mode2: npt.NDArray[np.complex128]


@dataclass
class ContinuousSchmidtMode(SchmidtMode):
    """A Schmidt mode for a continuous bipartite quantum system.

    Extends :class:`SchmidtMode` with the eigenmodes of both subsystems
    represented as :class:`~pyqentangle.core.wavefunctions.WaveFunction` objects
    (typically :class:`~pyqentangle.core.wavefunctions.InterpolatingWaveFunction` instances).

    Attributes:
        schmidt_coef (float): The Schmidt coefficient (singular value) for this mode.
        wavefunction1 (WaveFunction): Eigenmode wavefunction of the first subsystem.
        wavefunction2 (WaveFunction): Eigenmode wavefunction of the second subsystem.
    """

    wavefunction1: WaveFunction
    wavefunction2: WaveFunction
