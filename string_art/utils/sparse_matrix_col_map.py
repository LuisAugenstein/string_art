from scipy.sparse import csr_matrix, find
from typing import Callable
import numpy as np
from tqdm import tqdm


def sparse_matrix_col_map(f: Callable[[np.ndarray, int, np.ndarray], tuple[np.ndarray, np.ndarray]], A: csr_matrix, n_output_rows: int, use_tqdm: bool = False) -> csr_matrix:
    """
    Applies a function f to each column of a sparse matrix A and returns the new sparse matrix.

    Parameters
    f: (i,j,v) -> (new_i, new_v)  maps the j-th column i of the matrix A to a new column new_i with corresponding values 
    """
    _, n_cols = A.shape
    rows, cols, values = [], [], []
    iter = tqdm(range(n_cols)) if use_tqdm else range(n_cols)
    for col_index in iter:
        i, _, v = find(A[:, col_index])
        new_i, new_v = f(i, col_index, v)
        rows.append(new_i)
        cols.append(np.ones_like(new_i)*col_index)
        values.append(new_v)
    rows, cols, values = [np.concatenate(l) for l in [rows, cols, values]]
    return csr_matrix((values, (rows, cols)), shape=(n_output_rows, n_cols))
