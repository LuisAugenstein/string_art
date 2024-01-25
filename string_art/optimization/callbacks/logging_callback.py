from typing import Literal


class LoggingCallback:
    def __init__(self, n_edges: int) -> None:
        self.max_digits = len(str(n_edges))

    def choose_next_string(self, step: int, i_next_string: int | None, f_score: float | None) -> None:
        if i_next_string is None or f_score is None:
            print(f'{step}: edge-None')
            return
        n_padding = self.max_digits - len(str(i_next_string))
        print(f'{step}: edge-{str(i_next_string) + " " * n_padding}  {f_score:16.16f}')

    def switch_mode(self, new_mode: Literal['add', 'remove']) -> None:
        print(f'\n \tINFO: Switching mode to {new_mode} \n')
