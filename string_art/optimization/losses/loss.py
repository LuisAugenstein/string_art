from typing import Protocol, Literal
import numpy as np


class Loss(Protocol):
    def get_f_scores(self, x: np.ndarray, mode: Literal['add', 'remove'] = 'add') -> np.ndarray:
        """
        Parameters
        -
        x: np.shape([n_edges])   the current state of chosen edges. Values are either 0 (not chosen) or 1 (chosen).
        mode: 'add' | 'remove'   determines whether the f_scores are for adding or removing a string

        Returns
        -
        f_scores: np.shape([n_edges])   the f_scores for each edge. The lowest f_score indicates which edge to add/remove next.
        """
        ...
