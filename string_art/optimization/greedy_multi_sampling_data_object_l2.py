import numpy as np
from scipy.sparse import csr_matrix, find
import matplotlib.pyplot as plt
from string_art.optimization.losses import Loss
from typing import Literal

MIN_CIRCLE_LENGTH = 1


class GreedyMultiSamplingDataObjectL2:
    def __init__(self, loss: Loss, valid_edges_mask: np.ndarray):
        """
        img: np.shape([low_res, low_res])
        importance_map: np.shape([low_res, low_res])
        A_high_res: np.shape([high_res**2, n_edges])
        A_low_res: np.shape([low_res**2, n_edges])
        """
        self.loss = loss
        self.valid_edges_mask = valid_edges_mask

        self.x = np.zeros_like(valid_edges_mask, dtype=int)
        self.picked_edges_sequence = np.zeros(0, dtype=int)

    @property
    def removable_edge_indices(self) -> np.ndarray:
        return np.where(self.x == 1)[0]

    @property
    def addable_edge_indices(self) -> np.ndarray:
        return np.where((self.x == 0) & self.valid_edges_mask)[0]

    def find_best_string(self, mode: Literal['add', 'remove'] = 'add') -> tuple[np.ndarray, int]:
        f_scores = self.loss.get_f_scores(self.x, mode)
        candidate_edge_indices = self.addable_edge_indices if mode == 'add' else self.removable_edge_indices
        if candidate_edge_indices.size == 0:
            return None, None

        i_next_edge = candidate_edge_indices[np.argmin(f_scores[candidate_edge_indices])]
        f_score = f_scores[i_next_edge]
        return i_next_edge, f_score

    def choose_string_and_update(self, i, mode: Literal['add', 'remove'] = 'add'):
        if mode == 'remove':
            self.x[i] = 0
            self.picked_edges_sequence = self.picked_edges_sequence[self.picked_edges_sequence != i]
        else:
            self.x[i] = 1
            self.picked_edges_sequence = np.hstack((self.picked_edges_sequence.T, [i])).T
