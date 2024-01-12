from typing import Literal


class LoggingCallback:
    def __init__(self, n_edges: int) -> None:
        self.max_digits = len(str(n_edges))

    def choose_next_edge(self, step: int, i_next_edge: int | None, f_score: float | None) -> None:
        if i_next_edge is None or f_score is None:
            print(f'{step}: edge-None')
            return
        n_padding = self.max_digits - len(str(i_next_edge))
        print(f'{step}: edge-{str(i_next_edge) + " " * n_padding}  {f_score:16.16f}')

    def switch_mode(self, new_mode: Literal['add', 'remove']) -> None:
        print(f'\n \tINFO: Switching mode to {new_mode} \n')
