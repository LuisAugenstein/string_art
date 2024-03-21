from typing import Protocol, Literal
import torch


class Loss(Protocol):
    """All implementations of this protocol should be able to operate with cupy arrays as input."""

    def get_f_scores(self, x: torch.Tensor, mode: Literal['add', 'remove'] = 'add') -> torch.Tensor:
        """
        Parameters
        -
        x: torch.shape([n_strings])   the current state of chosen edges. Values are either 0 (not chosen) or 1 (chosen).
        mode: 'add' | 'remove'   determines whether the f_scores are for adding or removing a string

        Returns
        -
        f_scores: torch.shape([n_strings])   the f_scores for each edge. The lowest f_score indicates which edge to add/remove next.
        Note, that sqrt(f_scores / n_low_res_pixels) denotes the RMSE of the reconstruction after adding/removing the corresponding edge.
        """
        ...

    def update(self, i_next_string: int, mode: Literal['add', 'remove']) -> None:
        """
        Parameters
        -
        i_next_string: int   the index of the edge to add/remove next
        mode: 'add' | 'remove'   determines whether the edge is added or removed
        """
        ...
