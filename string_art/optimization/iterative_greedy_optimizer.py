import torch
from typing import Literal, Protocol
from string_art.optimization.losses import Loss
from string_art.optimization.callbacks import OptimizationCallback
from string_art.optimization.string_selection import StringSelection


class StringOptimizer(Protocol):
    def optimize(self) -> torch.Tensor:
        ...


class IterativeGreedyOptimizer:
    def __init__(self, loss: Loss, string_selection: StringSelection, callbacks: list[OptimizationCallback] = [], n_steps_max=100000) -> None:
        self.loss = loss
        self.string_selection = string_selection
        self.callbacks = callbacks
        self.n_steps_max = n_steps_max

    def optimize(self) -> torch.Tensor:
        """
        Parameters
        -
        callback: 
        """
        best_f_score = torch.inf
        switched_in_previous_iteration = False
        mode = 'add'

        for step in range(1, self.n_steps_max + 1):
            i_next_string, f_score = self.__find_best_string(mode)
            if f_score is None or f_score >= best_f_score:
                self.__callback_next_string(step, None, None)
                if switched_in_previous_iteration:
                    break
                switched_in_previous_iteration = True
                self.__switch_mode(mode)
                continue

            self.loss.update(i_next_string, mode)
            switched_in_previous_iteration = False
            self.__callback_next_string(step, i_next_string, f_score)
            self.string_selection.update(i_next_string, mode)
            best_f_score = f_score

        return self.string_selection.x

    def __find_best_string(self, mode: Literal['add', 'remove'] = 'add') -> tuple[int, float]:
        f_scores = self.loss.get_f_scores(mode)
        candidate_edge_indices = self.string_selection.get_selectable_strings(mode)
        if candidate_edge_indices.size == 0:
            return None, None
        i_min_fscore = torch.argmin(f_scores[candidate_edge_indices])
        i_next_edge = candidate_edge_indices[i_min_fscore]
        return i_next_edge.item(), f_scores[i_next_edge].item()

    def __callback_next_string(self, step: int, i_next_string: int | None, f_score: float | None) -> None:
        for c in self.callbacks:
            c.choose_next_string(step, i_next_string, f_score)

    def __switch_mode(self, mode: Literal['add', 'remove']) -> None:
        for c in self.callbacks:
            c.switch_mode(mode)
        return 'remove' if mode == 'add' else 'add'
