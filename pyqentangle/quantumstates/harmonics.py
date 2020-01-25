
import numpy as np
from scipy.special import hermite

disentangled_gaussian = lambda x1, x2: np.exp(-0.5 * (x1 * x1 + x2 * x2)) / np.sqrt(np.pi)

