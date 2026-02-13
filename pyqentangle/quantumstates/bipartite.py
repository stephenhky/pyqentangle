
import numpy as np


def create_singlet() -> np.ndarray:
    """Create a singlet state.

    Returns:
        Singlet state as a numpy array.
    """
    return np.array([[0., 1.],
                     [1., 0.]])

