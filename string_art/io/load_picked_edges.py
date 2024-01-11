import numpy as np
from scipy.sparse import csr_matrix
from string_art.optimization import IterativeGreedyOptimizer, OptimizedLoss, SimpleLoss


def load_picked_edges(img: np.ndarray, importance_map: np.ndarray, A_high_res: csr_matrix, A_low_res: csr_matrix, valid_edges_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # The SimpleLoss produces the same results as the OptimizedLoss, but is much slower.
    # loss = SimpleLoss(img, np.ones_like(importance_map), A_high_res, np.sqrt(A_low_res.shape[0]).astype(int))
    loss = OptimizedLoss(img, np.ones_like(importance_map), A_high_res, A_low_res)
    optimizer = IterativeGreedyOptimizer(loss, valid_edges_mask)
    return optimizer.optimize()
