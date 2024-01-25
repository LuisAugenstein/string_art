import numpy as np
from scipy.sparse import find, csc_matrix
from string_art.transformations import indices_1D_to_2D
from string_art.utils import map


def high_res_to_low_res_matrix(A_high_res: csc_matrix, low_res: int) -> csc_matrix:
    def col_mapping(col: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        i, v = col
        if i.shape[0] == 0:
            return i, v
        high_res = int(np.sqrt(A_high_res.shape[0]))
        # return high_res_to_low_res_indices(i, v, high_res, low_res)
        return high_res_to_low_res_indices_optimized(i, v, high_res, low_res)

    print(f'Compute A_low_res for low_res={low_res}')
    n_strings = A_high_res.shape[1]
    col_data = [(A_high_res[:, j].indices, A_high_res[:, j].data) for j in range(A_high_res.shape[1])]
    output_col_data = map(col_mapping, col_data)

    rows, cols, values = [], [], []
    for j, (i, v) in enumerate(output_col_data):
        rows.append(i)
        cols.append(np.ones_like(i)*j)
        values.append(v)
    rows, cols, values = [np.concatenate(x) for x in [rows, cols, values]]
    A_low_res = csc_matrix((values, (rows, cols)), shape=(low_res**2, n_strings))
    print(f'A_low_res.shape={A_low_res.shape[0]}x{A_low_res.shape[1]}')
    return A_low_res


def high_res_to_low_res_indices(high_res_indices: np.ndarray, high_res_values: np.ndarray, high_res: int, low_res: int) -> tuple[np.ndarray, np.ndarray]:
    scale = high_res//low_res
    img = np.zeros(high_res**2)
    img[high_res_indices] = high_res_values
    low_res_img = img.reshape((low_res, scale, low_res, scale)).mean(axis=(1, 3))
    k, j, v = find(low_res_img.T)
    i = j * low_res + k
    return i, v


def high_res_to_low_res_indices_optimized(high_res_indices: np.ndarray, high_res_values: np.ndarray, high_res: int, low_res: int) -> tuple[np.ndarray, np.ndarray]:
    scale = high_res//low_res
    xy = indices_1D_to_2D(high_res_indices, high_res, mode='row-col')  # [N,2]
    bbox_start = np.min(xy, axis=0)
    bbox_end = np.max(xy, axis=0)
    bbox_start -= bbox_start % scale
    bbox_end += scale - bbox_end % scale

    x, y = xy.T - bbox_start[:, None]
    bbox_end -= bbox_start
    img = np.zeros((bbox_end[0], bbox_end[1]))
    img[x, y] = high_res_values
    low_res_img = img.reshape((bbox_end[0]//scale, scale, bbox_end[1]//scale, scale)).mean(axis=(1, 3))
    k, j, v = find(low_res_img.T)
    j = j + bbox_start[0]//scale
    k = k + bbox_start[1]//scale
    i = j * low_res + k
    return i, v
