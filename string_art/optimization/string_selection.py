from typing import Literal
import torch


class StringSelection:

    def __init__(self, valid_edges_mask: torch.Tensor) -> None:
        """
        Parameters
        valid_edges_mask: np.shape([n_strings], dtype=bool)        False for excluding edges from the optimization.
        """
        self.__x = torch.zeros_like(valid_edges_mask, dtype=torch.int)
        self.__removable_edge_indices = torch.tensor([], dtype=torch.int)  # also indicates the order in which the edges were added
        self.__addable_edge_indices = torch.arange(self.__x.shape[0])[valid_edges_mask]

    @property
    def x(self) -> torch.Tensor:
        return self.__x

    def update(self, i_string: torch.Tensor, mode: Literal['add', 'remove']) -> None:
        """
        Parameters
        -
        i_string: torch.shape([], int)   index of the string to add or remove
        """
        if mode == 'add':
            self.__addable_edge_indices = self.__addable_edge_indices[self.__addable_edge_indices != i_string]
            self.__removable_edge_indices = torch.concat([self.__removable_edge_indices, torch.Tensor([i_string])])
            self.__x[i_string] = 1
        elif mode == 'remove':
            self.__addable_edge_indices = torch.concat([self.__addable_edge_indices, torch.Tensor([i_string])])
            self.__removable_edge_indices = self.__removable_edge_indices[self.__removable_edge_indices != i_string]
            self.__x[i_string] = 0

    def get_selectable_strings(self, mode: Literal['add', 'remove']) -> torch.Tensor:
        return self.__addable_edge_indices if mode == 'add' else self.__removable_edge_indices
