import numpy as np
from typing import Literal
from string_art.optimization.losses import Loss
from string_art.optimization.callbacks import OptimizationCallback, LoggingCallback


class IterativeGreedyOptimizer:
    def __init__(self, loss: Loss, valid_edges_mask: np.ndarray) -> None:
        self.loss = loss
        self.valid_edges_mask = valid_edges_mask
        self.x = np.zeros_like(valid_edges_mask, dtype=int)

    @property
    def removable_edge_indices(self) -> np.ndarray:
        return np.where(self.x == 1)[0]

    @property
    def addable_edge_indices(self) -> np.ndarray:
        return np.where((self.x == 0) & self.valid_edges_mask)[0]

    def optimize(self, callback: OptimizationCallback | None = None, n_steps=1000) -> np.ndarray:
        """
        Parameters
        -
        callback: 
        """
        if callback is None:
            callback = LoggingCallback(self.x.size)
        best_f_score = np.inf
        mode = 'add'
        switched = False

        for step in range(1, n_steps + 1):
            i_next_edge, f_score = self.__find_best_string(mode)
            if f_score is None or f_score >= best_f_score:
                callback.choose_next_edge(step, None, None)
                if switched:
                    break
                switched = True
                new_mode = 'remove' if mode == 'add' else 'add'
                callback.switch_mode(new_mode)
                continue

            switched = False
            callback.choose_next_edge(step, i_next_edge, f_score)
            self.x[i_next_edge] = 1 if mode == 'add' else 0
            best_f_score = f_score

        return self.x

    def __find_best_string(self, mode: Literal['add', 'remove'] = 'add') -> tuple[int, np.ndarray]:
        f_scores = self.loss.get_f_scores(self.x, mode)
        candidate_edge_indices = self.addable_edge_indices if mode == 'add' else self.removable_edge_indices
        if candidate_edge_indices.size == 0:
            return None, None

        i_next_edge = candidate_edge_indices[np.argmin(f_scores[candidate_edge_indices])]
        f_score = f_scores[i_next_edge]
        return i_next_edge, f_score
