import numpy as np
from itertools import combinations


def get_edges(n_pins: int) -> np.ndarray:
    """
    Returns
    edges: np.shape([n_edges, 2]) (i,j) where i,j are the indices of the pins
    """
    return np.array(list(combinations(range(n_pins), 2)))
