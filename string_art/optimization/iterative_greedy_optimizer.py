import numpy as np
import cupy as cp
from typing import Literal
from string_art.optimization.losses import Loss
from string_art.optimization.callbacks import OptimizationCallback, LoggingCallback
from string_art.optimization.string_selection import StringSelection
import math


class IterativeGreedyOptimizer:
    def __init__(self, loss: Loss, string_selection: StringSelection) -> None:
        self.loss = loss
        self.string_selection = string_selection

    def optimize(self, callbacks: list[OptimizationCallback] = [], n_steps=1000) -> np.ndarray:
        """
        Parameters
        -
        callback: 
        """
        if len(callbacks) == 0:
            callbacks = [LoggingCallback(self.string_selection.x.size)]
        best_f_score = math.inf
        switched_in_previous_iteration = False
        mode = 'add'

        for step in range(1, n_steps + 1):
            i_next_string, f_score = self.__find_best_string(mode)
            if f_score is None or f_score >= best_f_score:
                [c.choose_next_edge(step, None, None) for c in callbacks]
                if switched_in_previous_iteration:
                    break
                switched_in_previous_iteration = True
                mode = 'remove' if mode == 'add' else 'add'
                [c.switch_mode(mode) for c in callbacks]
                continue

            self.loss.update(i_next_string, mode)
            switched_in_previous_iteration = False
            [c.choose_next_edge(step, i_next_string, f_score) for c in callbacks]
            self.string_selection.update(i_next_string, mode)
            best_f_score = f_score

        return self.string_selection.x

    def __find_best_string(self, mode: Literal['add', 'remove'] = 'add') -> tuple[int, float]:
        f_scores = self.loss.get_f_scores(mode)
        candidate_edge_indices = self.string_selection.get_selectable_strings(mode)
        if candidate_edge_indices.size == 0:
            return None, None
        i_min_fscore = np.argmin(f_scores[candidate_edge_indices])
        i_next_edge = candidate_edge_indices[i_min_fscore.get() if isinstance(i_min_fscore, cp.ndarray) else i_min_fscore]
        f_score = f_scores[i_next_edge].get() if isinstance(f_scores[i_next_edge], cp.ndarray) else f_scores[i_next_edge]
        return i_next_edge, float(f_score)
