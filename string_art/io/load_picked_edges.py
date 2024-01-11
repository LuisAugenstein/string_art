from string_art.optimization import optimize_strings_greedy_multi_sampling
import numpy as np
from scipy.sparse import csr_matrix


def load_picked_edges(img: np.ndarray, importance_map: np.ndarray, A_high_res: csr_matrix, A_low_res: csr_matrix, valid_edges_mask: np.ndarray):
    return optimize_strings_greedy_multi_sampling(img, importance_map, A_high_res, A_low_res, valid_edges_mask)
