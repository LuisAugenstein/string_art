import numpy as np
from typing import Literal
from string_art.optimization.losses import Loss


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

    def optimize(self, n_steps=1000) -> tuple[np.ndarray, np.ndarray]:
        best_f_score = np.inf
        mode = 'add'
        switched = False

        for step in range(1, n_steps + 1):
            i_next_edge, f_score = self.__find_best_string(mode)
            if f_score is None or f_score >= best_f_score:
                print(f'{step}: edge-None')
                if switched:
                    break
                switched = True
                mode = self.__switch_mode(mode)
                continue

            switched = False
            self.__print_loss(step, i_next_edge, f_score)
            self.x[i_next_edge] = 1 if mode == 'add' else 0
            best_f_score = f_score

        return self.x

    def __find_best_string(self, mode: Literal['add', 'remove'] = 'add') -> tuple[np.ndarray, int]:
        f_scores = self.loss.get_f_scores(self.x, mode)
        candidate_edge_indices = self.addable_edge_indices if mode == 'add' else self.removable_edge_indices
        if candidate_edge_indices.size == 0:
            return None, None

        i_next_edge = candidate_edge_indices[np.argmin(f_scores[candidate_edge_indices])]
        f_score = f_scores[i_next_edge]
        return i_next_edge, f_score

    def __print_loss(self, step: int, i_next_edge: int, f_score: float):
        n_padding = len(str(self.valid_edges_mask.size)) - len(str(i_next_edge))
        print(f'{step}: edge-{str(i_next_edge) + ' ' * n_padding}  {f_score:16.16f}')

    def __switch_mode(self, mode: Literal['add', 'remove'] = 'add') -> Literal['add', 'remove']:
        new_mode = 'remove' if mode == 'add' else 'add'
        print(f'\n \tINFO: Switching mode to {new_mode} \n')
        return new_mode
