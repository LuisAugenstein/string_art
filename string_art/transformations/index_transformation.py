import numpy as np


def indices_2D_to_1D(x: np.ndarray, y: np.ndarray, domain_width: float) -> np.ndarray:
    return y * domain_width + x


def indices_1D_to_2D(i: np.ndarray, domain_width: float) -> np.ndarray:
    return np.vstack([i % domain_width, i // domain_width])
