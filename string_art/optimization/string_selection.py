import numpy as np
from typing import Literal


class StringSelection:

    def __init__(self, valid_edges_mask: np.ndarray) -> None:
        """
        Parameters
        valid_edges_mask: np.shape([n_strings], dtype=bool)        False for excluding edges from the optimization.
        """
        self.__x = np.zeros_like(valid_edges_mask, dtype=int)
        self.__removable_edge_indices = np.array([], dtype=int)  # also indicates the order in which the edges were added
        self.__addable_edge_indices = np.arange(self.__x.shape[0])[valid_edges_mask]

    @property
    def x(self) -> np.ndarray:
        return self.__x

    def update(self, i_string: int, mode: Literal['add', 'remove']) -> None:
        if mode == 'add':
            self.__addable_edge_indices = self.__addable_edge_indices[self.__addable_edge_indices != i_string]
            self.__removable_edge_indices = np.append(self.__removable_edge_indices, i_string)
            self.__x[i_string] = 1
        elif mode == 'remove':
            self.__addable_edge_indices = np.append(self.__addable_edge_indices, i_string)
            self.__removable_edge_indices = self.__removable_edge_indices[self.__removable_edge_indices != i_string]
            self.__x[i_string] = 0

    def get_selectable_strings(self, mode: Literal['add', 'remove']) -> np.ndarray:
        return self.__addable_edge_indices if mode == 'add' else self.__removable_edge_indices
