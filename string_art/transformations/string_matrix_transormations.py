import numpy as np
from string_art.entities import String
from scipy.sparse import csr_matrix, find
from string_art.transformations.index_transformation import indices_1D_to_2D, indices_2D_to_1D


def strings_to_sparse_matrix(strings: list[String], resolution: int) -> csr_matrix:
    """
    Returns
    -
    A: np.shape([n_pixels, n_edges])    values between 0 and 1 indicate how much a pixel i is darkened if edge j is active.
    """
    n_pixels, n_edges = resolution**2, len(strings)
    rows, cols, values = [], [], []
    for j, string in enumerate(strings):
        x, y, v = string.T
        i = indices_2D_to_1D(x, y, resolution)
        rows.append(i)
        cols.append([j]*v.shape[0])
        values.append(v)
    rows, cols, values = [np.concatenate(l) for l in [rows, cols, values]]
    return csr_matrix((values, (rows, cols)), shape=(n_pixels, n_edges))


def sparse_matrix_to_strings(A: csr_matrix) -> list[String]:
    n_pixels, n_edges = A.shape
    resolution = int(np.sqrt(n_pixels))
    strings = []
    for j in range(n_edges):
        i, _, v = find(A[:, j])
        x, y = indices_1D_to_2D(i, resolution)
        strings.append(np.vstack([x, y, v]).T)
    return strings
