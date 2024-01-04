from string_art.optimization import optimize_strings_greedy_multi_sampling
import numpy as np
from scipy.sparse import csr_matrix


def load_picked_edges(img: np.ndarray, super_sampling_factor: int, min_angle: float, n_pins: int, A_high_res: csr_matrix, A_low_res: csr_matrix, fabricable_edges=None, importance_map=None):
    return optimize_strings_greedy_multi_sampling(img, super_sampling_factor, min_angle, n_pins, A_high_res, A_low_res, fabricable_edges, importance_map)
