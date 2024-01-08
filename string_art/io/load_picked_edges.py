from string_art.optimization import optimize_strings_greedy_multi_sampling
import numpy as np
from scipy.sparse import csr_matrix
from string_art.transformations import PinEdgeTransformer


def load_picked_edges(img: np.ndarray, importance_map: np.ndarray, A_high_res: csr_matrix, A_low_res: csr_matrix, pin_edge_transformer: PinEdgeTransformer, min_angle: float):
    return optimize_strings_greedy_multi_sampling(img, importance_map, A_high_res, A_low_res, pin_edge_transformer, min_angle)
