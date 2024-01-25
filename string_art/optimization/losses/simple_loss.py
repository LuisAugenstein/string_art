import numpy as np
from scipy.sparse import csc_matrix
from typing import Literal
from string_art.optimization.losses.multi_sample_correspondence_map import multi_sample_correspondence_map
from string_art.api import get_np_array_module


class SimpleLoss:
    def __init__(self, img: np.ndarray, importance_map: np.ndarray, A_high_res: csc_matrix, low_res: int) -> None:
        self.xp, self.xipy = get_np_array_module(img)
        self.b_native_res = img.flatten()
        self.importance_map = importance_map.flatten()
        self.A_high_res = A_high_res
        self.low_res = low_res
        high_res = int(self.xp.sqrt(A_high_res.shape[0]))
        self.B = self.xipy.sparse.csr_matrix(multi_sample_correspondence_map(self.low_res, high_res))
        self.__x = self.xp.zeros(self.A_high_res.shape[1], dtype=self.xp.float32)

    def update(self, i_next_string: int, mode: Literal['add', 'remove'] = 'add') -> None:
        self.__x[i_next_string] = 1 if mode == 'add' else 0

    def get_f_scores(self, mode: Literal['add', 'remove'] = 'add') -> np.ndarray:
        xp = self.xp
        _, n_strings = xp.sqrt(self.A_high_res.shape[0]).astype(int), self.A_high_res.shape[1]
        f_scores = xp.ones(n_strings) * xp.inf
        candidate_edges = xp.where(self.__x == 0)[0] if mode == 'add' else xp.where(self.__x == 1)[0]
        for k in candidate_edges:
            x_current = self.__x.copy()
            x_current[k] = 1
            Ax = (self.A_high_res @ x_current).squeeze()
            CAx = xp.clip(Ax, 0, 1)
            BCAx = self.B @ CAx
            if mode == 'add':
                f_scores[k] = xp.sum((self.importance_map * (self.b_native_res - BCAx))**2)
            else:
                f_scores[k] = xp.sum((self.importance_map * (self.b_native_res + BCAx))**2)
        return f_scores
