
import numpy as np
import numpy.typing as npt


def create_singlet() -> npt.NDArray[np.complex128]:
    """Create a two-qubit singlet state tensor.

    Returns the bipartite state tensor for the singlet state
    :math:`|\\Psi^+\\rangle = \\frac{1}{\\sqrt{2}}(|01\\rangle + |10\\rangle)`,
    where element ``[i, j]`` is the coefficient of :math:`|ij\\rangle`.

    Note: the returned tensor is **not** normalized; multiply by
    :math:`1/\\sqrt{2}` to obtain the normalized singlet state.

    Returns:
        numpy.ndarray: A ``(2, 2)`` array representing the singlet state tensor.
    """
    return np.array([[0., 1.],
                     [1., 0.]])
