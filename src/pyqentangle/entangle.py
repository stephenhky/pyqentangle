
from typing import Union, Literal, Generator, Optional

import numpy as np
import numpy.typing as npt

from .core.schmidt import schmidt_decomposition
from .core.continuous import continuous_schmidt_decomposition
from .core.wavefunctions import WaveFunction, AnalyticMultiDimWaveFunction
from .schemas import DiscreteSchmidtMode, ContinuousSchmidtMode


class DiscreteSchmidtDecomposer:
    """Compute and store the Schmidt decomposition of a discrete bipartite quantum state.

    Given a 2-D tensor whose element ``tensor[i, j]`` is the coefficient of the ket
    :math:`|ij\\rangle`, this class performs the Schmidt decomposition and exposes the
    resulting modes through :meth:`modes` and :meth:`mode_iterator`.

    The decomposition can be computed eagerly (default) or lazily (deferred until the
    first access of the results).
    """

    def __init__(
            self,
            tensor: Union[npt.NDArray[np.complex128], npt.NDArray[np.float64]],
            lazy: bool = False,
            approach: Literal["tensornetwork", "numpy"] = "tensornetwork"
    ):
        """Initialize the decomposer with a bipartite state tensor.

        Args:
            tensor (numpy.ndarray): 2-D array describing the bipartite state, where
                ``tensor[i, j]`` is the coefficient of :math:`|ij\\rangle`.
            lazy (bool, optional): If ``True``, defer computation until the results are
                first accessed. Defaults to ``False``.
            approach (str, optional): Backend to use for the SVD. Either ``'tensornetwork'``
                or ``'numpy'``. Defaults to ``'tensornetwork'``.
        """
        self._tensor = tensor
        self._lazy = lazy
        self._approach = approach

        self._calculated = False
        if not self._lazy:
            self._compute()

    def _compute(self) -> None:
        """Run the Schmidt decomposition and store the results internally."""
        raw_decomposed_results = schmidt_decomposition(self._tensor, self._approach)
        self._results = [
            DiscreteSchmidtMode(schmidt_coef=item[0], mode1=item[1], mode2=item[2])
            for item in raw_decomposed_results
        ]
        self._calculated = True

    def modes(self) -> list[DiscreteSchmidtMode]:
        """Return all Schmidt modes.

        If the decomposer was created in lazy mode and the decomposition has not yet
        been computed, it is computed on first call.

        Returns:
            list[DiscreteSchmidtMode]: List of Schmidt modes, each containing a Schmidt
            coefficient and the corresponding eigenvectors for both subsystems.
        """
        if not self._calculated:
            # lazy mode, not calculated previously
            self._compute()
        return self._results

    def mode_iterator(self) -> Generator[DiscreteSchmidtMode, None, None]:
        """Iterate over all Schmidt modes one at a time.

        If the decomposer was created in lazy mode and the decomposition has not yet
        been computed, it is computed on first call.

        Yields:
            DiscreteSchmidtMode: Each Schmidt mode in turn.
        """
        if not self._calculated:
            # lazy mode, not calculated previously
            self._compute()
        for result in self._results:
            yield result

    @property
    def tensor(self) -> Union[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
        """The bipartite state tensor used for the decomposition.

        Returns:
            numpy.ndarray: The original 2-D bipartite state tensor.
        """
        return self._tensor

    @property
    def calculated(self) -> bool:
        """Whether the Schmidt decomposition has been computed.

        Returns:
            bool: ``True`` if the decomposition has been computed, ``False`` otherwise.
        """
        return self._calculated

    @property
    def approach(self) -> Literal["tensornetwork", "numpy"]:
        """The computational backend used for the SVD.

        Returns:
            str: Either ``'tensornetwork'`` or ``'numpy'``.
        """
        return self._approach


class ContinuousSchmidtDecomposer:
    """Compute and store the Schmidt decomposition of a continuous bipartite quantum state.

    Given a bipartite wavefunction :math:`\\psi(x_1, x_2)` defined over the rectangular
    domain :math:`[x_{1,\\text{lo}}, x_{1,\\text{hi}}] \\times [x_{2,\\text{lo}}, x_{2,\\text{hi}}]`,
    this class discretizes the wavefunction on a grid and performs the Schmidt decomposition.
    The resulting modes are returned as :class:`~pyqentangle.schemas.schemas.ContinuousSchmidtMode`
    objects whose wavefunctions are :class:`~pyqentangle.core.wavefunctions.InterpolatingWaveFunction`
    instances.

    The decomposition can be computed eagerly (default) or lazily (deferred until the
    first access of the results).
    """

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
        """Initialize the decomposer with a continuous bipartite wavefunction.

        Args:
            bipartite_wavefunction (callable or WaveFunction): The bipartite wavefunction
                :math:`\\psi(x_1, x_2)`. If not already a :class:`~pyqentangle.core.wavefunctions.WaveFunction`
                instance, it is wrapped in an :class:`~pyqentangle.core.wavefunctions.AnalyticMultiDimWaveFunction`.
            x1_lo (float): Lower bound of the first subsystem coordinate :math:`x_1`.
            x1_hi (float): Upper bound of the first subsystem coordinate :math:`x_1`.
            x2_lo (float): Lower bound of the second subsystem coordinate :math:`x_2`.
            x2_hi (float): Upper bound of the second subsystem coordinate :math:`x_2`.
            nb_x1 (int, optional): Number of grid points along :math:`x_1`. Defaults to 100.
            nb_x2 (int, optional): Number of grid points along :math:`x_2`. Defaults to 100.
            keep (int, optional): Number of Schmidt modes (with the largest coefficients) to
                retain. If ``None``, all modes up to ``min(nb_x1, nb_x2)`` are kept.
                Defaults to ``None``.
            lazy (bool, optional): If ``True``, defer computation until the results are
                first accessed. Defaults to ``False``.
            approach (str, optional): Backend to use for the SVD. Either ``'tensornetwork'``
                or ``'numpy'``. Defaults to ``'tensornetwork'``.
        """
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
        """Discretize the wavefunction and run the Schmidt decomposition, storing the results."""
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
        """Return all Schmidt modes.

        If the decomposer was created in lazy mode and the decomposition has not yet
        been computed, it is computed on first call.

        Returns:
            list[ContinuousSchmidtMode]: List of Schmidt modes, each containing a Schmidt
            coefficient and the corresponding interpolating wavefunctions for both subsystems.
        """
        if not self._calculated:
            # lazy mode, not calculated previously
            self._compute()
        return self._results

    def mode_iterator(self) -> Generator[ContinuousSchmidtMode, None, None]:
        """Iterate over all Schmidt modes one at a time.

        If the decomposer was created in lazy mode and the decomposition has not yet
        been computed, it is computed on first call.

        Yields:
            ContinuousSchmidtMode: Each Schmidt mode in turn.
        """
        if not self._calculated:
            # lazy mode, not calculated previously
            self._compute()
        for result in self._results:
            yield result

    @property
    def bipartite_wavefuncion(self) -> WaveFunction:
        """The bipartite wavefunction used for the decomposition.

        Returns:
            WaveFunction: The (possibly wrapped) bipartite wavefunction.
        """
        return self._bipartitle_wavefunction

    @property
    def x1_lo(self) -> float:
        """Lower bound of the first subsystem coordinate.

        Returns:
            float: Lower bound :math:`x_{1,\\text{lo}}`.
        """
        return self._x1_lo

    @property
    def x1_hi(self) -> float:
        """Upper bound of the first subsystem coordinate.

        Returns:
            float: Upper bound :math:`x_{1,\\text{hi}}`.
        """
        return self._x1_hi

    @property
    def x2_lo(self) -> float:
        """Lower bound of the second subsystem coordinate.

        Returns:
            float: Lower bound :math:`x_{2,\\text{lo}}`.
        """
        return self._x2_lo

    @property
    def x2_hi(self) -> float:
        """Upper bound of the second subsystem coordinate.

        Returns:
            float: Upper bound :math:`x_{2,\\text{hi}}`.
        """
        return self._x2_hi

    @property
    def nb_x1(self) -> int:
        """Number of grid points along the first subsystem coordinate.

        Returns:
            int: Number of grid points along :math:`x_1`.
        """
        return self._nb_x1

    @property
    def nb_x2(self) -> int:
        """Number of grid points along the second subsystem coordinate.

        Returns:
            int: Number of grid points along :math:`x_2`.
        """
        return self._nb_x2

    @property
    def calculated(self) -> bool:
        """Whether the Schmidt decomposition has been computed.

        Returns:
            bool: ``True`` if the decomposition has been computed, ``False`` otherwise.
        """
        return self._calculated

    @property
    def approach(self) -> Literal["tensornetwork", "numpy"]:
        """The computational backend used for the SVD.

        Returns:
            str: Either ``'tensornetwork'`` or ``'numpy'``.
        """
        return self._approach
