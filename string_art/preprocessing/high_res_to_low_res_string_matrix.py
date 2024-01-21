import numpy as np
from scipy.sparse import find, csr_matrix
from tqdm import tqdm
from string_art.transformations import indices_1D_to_2D
from string_art.utils import sparse_matrix_col_map


def high_res_to_low_res_matrix(A_high_res: csr_matrix, low_res: int) -> csr_matrix:
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
