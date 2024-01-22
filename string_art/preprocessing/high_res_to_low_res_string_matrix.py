import numpy as np
from scipy.sparse import find, csc_matrix
from tqdm import tqdm
from string_art.transformations import indices_1D_to_2D
from typing import Callable


def high_res_to_low_res_matrix(A_high_res: csc_matrix, low_res: int) -> csc_matrix:
    def col_mapping(i, _, v): return high_res_to_low_res_indices_optimized(i, v, A_high_res.shape[0], low_res)

    print(f'Compute A_low_res for low_res={low_res}')
    A_low_res = sparse_matrix_col_map(col_mapping, A_high_res, low_res**2, use_tqdm=True)
    print(f'A_low_res.shape={A_low_res.shape[0]}x{A_low_res.shape[1]}')
    return A_low_res


def high_res_to_low_res_indices(high_res_indices: np.ndarray, high_res_values: np.ndarray, high_res_squared: int, low_res: int) -> tuple[np.ndarray, np.ndarray]:
    high_res = int(np.sqrt(high_res_squared))
    scale = high_res//low_res
    img = np.zeros(high_res_squared)
    img[high_res_indices] = high_res_values
    low_res_img = img.reshape((low_res, scale, low_res, scale)).mean(axis=(1, 3))
    k, j, v = find(low_res_img.T)
    i = j * low_res + k
    return i, v


def high_res_to_low_res_indices_optimized(high_res_indices: np.ndarray, high_res_values: np.ndarray, high_res_squared: int, low_res: int) -> tuple[np.ndarray, np.ndarray]:
    high_res = int(np.sqrt(high_res_squared))
    scale = high_res//low_res

    x, y = indices_1D_to_2D(high_res_indices, high_res, mode='row-col').T
    bbox_start = np.array([min(x[0], x[-1]), min(y[0], y[-1])])
    reduce_x, reduce_y = bbox_start - bbox_start % scale
    img = np.zeros((high_res-reduce_x, high_res-reduce_y))
    img[x - reduce_x, y - reduce_y] = high_res_values
    low_res_img = img.reshape((low_res-reduce_x//scale, scale, low_res - reduce_y//scale, scale)).mean(axis=(1, 3))
    k, j, v = find(low_res_img.T)
    j = j + reduce_x//scale
    k = k + reduce_y//scale
    i = j * low_res + k
    return i, v


def sparse_matrix_col_map(f: Callable[[np.ndarray, int, np.ndarray], tuple[np.ndarray, np.ndarray]], A: csc_matrix, n_output_rows: int, use_tqdm: bool = False) -> csc_matrix:
    """
    Applies a function f to each column of a sparse matrix A and returns the new sparse matrix.

    Parameters
    f: (i,j,v) -> (new_i, new_v)  maps the j-th column i of the matrix A to a new column new_i with corresponding values 
    """
    _, n_cols = A.shape
    rows, cols, values = [], [], []
    iter = tqdm(range(n_cols)) if use_tqdm else range(n_cols)
    for col_index in iter:
        i, v = A[:, col_index].indices, A[:, col_index].data
        new_i, new_v = f(i, col_index, v)
        rows.append(new_i)
        cols.append(np.ones_like(new_i)*col_index)
        values.append(new_v)
    rows, cols, values = [np.concatenate(l) for l in [rows, cols, values]]
    return csc_matrix((values, (rows, cols)), shape=(n_output_rows, n_cols))
