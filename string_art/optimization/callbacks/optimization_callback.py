from typing import Protocol, Literal


class OptimizationCallback(Protocol):
    def choose_next_string(self, step: int, i_next_edge: int | None, f_score: float | None) -> None:
        """
        Parameters
        -
        step:         the current step of the iteration
        i_next_edge:  the index of the next edge to be added or removed which was chosen at the current step
        f_score:      the f-score after adding/removing i_next_edge
        """
        ...

    def switch_mode(self, new_mode: Literal['add', 'remove']) -> None:
        ...
