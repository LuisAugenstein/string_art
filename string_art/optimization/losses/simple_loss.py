import numpy as np
import cupy as cp
import scipy.sparse as scipy_sparse
import cupy.sparse as cp_sparse
from typing import Literal
from string_art.optimization.losses.multi_sample_correspondence_map import multi_sample_correspondence_map


class SimpleLoss:
    def __init__(self, img: np.ndarray, importance_map: np.ndarray, A_high_res: scipy_sparse.csc_matrix, low_res: int) -> None:
        self.b_native_res = cp.array(img.flatten())
        self.importance_map = cp.array(importance_map.flatten())
        self.A_high_res = cp_sparse.csc_matrix(A_high_res)
        self.low_res = low_res
        self.B = multi_sample_correspondence_map(self.low_res, high_res=int(np.sqrt(self.A_high_res.shape[0])))
        self.__x = cp.zeros(self.A_high_res.shape[1], dtype=cp.float32)

    def update(self, i_next_string: int, mode: Literal['add', 'remove'] = 'add') -> None:
        self.__x[i_next_string] = 1 if mode == 'add' else 0

    def get_f_scores(self, mode: Literal['add', 'remove'] = 'add') -> cp.ndarray:
        _, n_strings = cp.sqrt(self.A_high_res.shape[0]).astype(int), self.A_high_res.shape[1]
        f_scores = cp.ones(n_strings) * cp.inf
        candidate_edges = cp.where(self.__x == 0)[0] if mode == 'add' else cp.where(self.__x == 1)[0]
        for k in candidate_edges:
            x_current = self.__x.copy()
            x_current[k] = 1
            Ax = (self.A_high_res @ x_current).squeeze()
            CAx = cp.clip(Ax, 0, 1)
            BCAx = self.B @ CAx
            if mode == 'add':
                f_scores[k] = cp.sum((self.importance_map * (self.b_native_res - BCAx))**2)
            else:
                f_scores[k] = cp.sum((self.importance_map * (self.b_native_res + BCAx))**2)
        return f_scores
