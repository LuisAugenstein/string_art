import numpy as np
from typing import Literal
from string_art.optimization.multi_sample_correspondence_map import multi_sample_correspondence_map
from scipy.sparse import csr_matrix


class SimpleLoss:
    def __init__(self, img: np.ndarray, importance_map: np.ndarray, A_high_res: csr_matrix, low_res: int):
        self.b_native_res = img.flatten()
        self.importance_map = importance_map.flatten()
        self.A_high_res = A_high_res
        self.low_res = low_res

    def get_f_scores(self, x: np.ndarray, mode: Literal['add', 'remove'] = 'add') -> np.ndarray:
        high_res, n_edges = np.sqrt(self.A_high_res.shape[0]).astype(int), self.A_high_res.shape[1]
        B = multi_sample_correspondence_map(self.low_res, high_res)
        f_scores = np.ones(n_edges) * np.inf
        unset_edges = np.where(x == 0)[0]
        for k in unset_edges:
            x_current = x.copy()
            x_current[k] = 1
            Ax = (self.A_high_res @ x_current).squeeze()
            CAx = np.clip(Ax, 0, 1)
            BCAx = B @ CAx
            if mode == 'add':
                f_scores[k] = np.sum((self.importance_map * (self.b_native_res - BCAx))**2)
            else:
                f_scores[k] = np.sum((self.importance_map * (self.b_native_res + BCAx))**2)
        return f_scores
