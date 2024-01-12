import os
import numpy as np
from scipy.sparse import csr_matrix
from string_art.optimization import IterativeGreedyOptimizer, OptimizedLoss, SimpleLoss, OptimizationCallback
from string_art.io.root_path import get_project_dir


def load_picked_edges(name_of_the_run: str, image: np.ndarray, importance_map: np.ndarray, A_high_res: csr_matrix, A_low_res: csr_matrix, valid_edges_mask: np.ndarray, callbacks: list[OptimizationCallback]) -> np.ndarray:
    project_dir = get_project_dir(name_of_the_run)
    x_path = f'{project_dir}/x.npy'
    if os.path.exists(x_path):
        return np.load(x_path)

    # The SimpleLoss produces the same results as the OptimizedLoss, but is much slower.
    # loss = SimpleLoss(img, np.ones_like(importance_map), A_high_res, np.sqrt(A_low_res.shape[0]).astype(int))
    loss = OptimizedLoss(image, np.ones_like(importance_map), A_high_res, A_low_res)
    optimizer = IterativeGreedyOptimizer(loss, valid_edges_mask)
    x = optimizer.optimize(callbacks)
    np.save(x_path, x)
    return x
