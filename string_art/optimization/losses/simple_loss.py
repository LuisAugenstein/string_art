from typing import Literal
from string_art.optimization.losses.multi_sample_correspondence_map import multi_sample_correspondence_map
import torch
import numpy as np


class SimpleLoss:
    def __init__(self, img: torch.Tensor, importance_map: torch.Tensor, A_high_res: torch.Tensor) -> None:
        """
        Parameters
        -
        img:            torch.shape([low_res, low_res]) grayscale image with values between 0 and 1
        importance_map: torch.shape([low_res, low_res]) scalars between 0 and 1 to weight the reconstruction importance of different image regions
        A_high_res:     torch.shape([high_res**2, n_strings])
        """
        low_res = img.shape[0]
        high_res = int(np.sqrt(A_high_res.shape[0]))
        n_strings = A_high_res.shape[1]

        self.b_native_res = img.flatten()  # [low_res**2]
        self.importance_map = importance_map.flatten()  # [low_res**2]
        self.A_high_res = A_high_res.to_sparse_csr()
        self.B = multi_sample_correspondence_map(low_res, high_res).to_sparse_csr()  # csr_matrix ?
        self.__x = torch.zeros(n_strings)

    def update(self, i_next_string: int, mode: Literal['add', 'remove'] = 'add') -> None:
        self.__x[i_next_string] = 1 if mode == 'add' else 0

    def get_f_scores(self, mode: Literal['add', 'remove'] = 'add') -> torch.Tensor:
        f_scores = torch.ones_like(self.__x) * torch.inf
        candidate_edges = torch.where(self.__x == 0)[0] if mode == 'add' else torch.where(self.__x == 1)[0]
        for i, k in enumerate(candidate_edges):
            x_current = self.__x.clone()
            x_current[k] = 1
            Ax = (self.A_high_res @ x_current).squeeze()
            CAx = torch.clip(Ax, 0, 1)  # high_res**2
            BCAx = self.B @ CAx
            if mode == 'add':
                f_scores[k] = torch.sum((self.importance_map * (self.b_native_res - BCAx))**2)
            else:
                f_scores[k] = torch.sum((self.importance_map * (self.b_native_res + BCAx))**2)
        return f_scores.numpy()
