import numpy as np
from string_art.entities import String
from scipy.sparse import csc_matrix
from string_art.transformations.index_transformation import indices_1D_to_2D, indices_2D_to_1D
import torch


def strings_to_sparse_matrix(strings: list[String], resolution: int) -> torch.Tensor:
    """
    Returns
    -
    A: torch.shape([n_pixels, n_strings])    values between 0 and 1 indicate how much a pixel i is darkened if edge j is active.
    """
    n_pixels, n_strings = resolution**2, len(strings)
    rows, cols, values = [], [], []
    for j, string in enumerate(strings):
        x, y, v = string
        i = indices_2D_to_1D(x, y, resolution)
        rows.append(i)
        cols.append(torch.ones_like(v)*j)
        values.append(v)
    rows, cols, values = [torch.concatenate(l) for l in [rows, cols, values]]
    indices = torch.stack([rows, cols])
    return torch.sparse_coo_tensor(indices, values, size=(n_pixels, n_strings), is_coalesced=True)


def sparse_matrix_to_strings(A: csc_matrix) -> list[String]:
    n_pixels, n_strings = A.shape
    resolution = int(np.sqrt(n_pixels))
    strings = []
    for j in range(n_strings):
        i, v = A[:, j].indices, A[:, j].data
        x, y = indices_1D_to_2D(i, resolution).T
        strings.append((x, y, v))
    return strings
