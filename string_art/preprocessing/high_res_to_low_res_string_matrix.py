import numpy as np
from scipy.sparse import find, csc_matrix
from string_art.transformations import indices_1D_to_2D
from tqdm import tqdm
from string_art.utils import parallel_map


def high_res_to_low_res_matrix(A_high_res: csc_matrix, low_res: int) -> csc_matrix:
    def col_mapping(col: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        i, v = col
        if i.shape[0] == 0:
            return i, v
        # return high_res_to_low_res_indices(i, v, A_high_res.shape[0], low_res)
        return high_res_to_low_res_indices_optimized(i, v, A_high_res.shape[0], low_res)

    print(f'Compute A_low_res for low_res={low_res}')
    n_strings = A_high_res.shape[1]
    col_data = [(A_high_res[:, j].indices, A_high_res[:, j].data) for j in range(A_high_res.shape[1])]
    # output_col_data = [col_mapping(col) for col in tqdm(col_data)]
    output_col_data = parallel_map(col_mapping, col_data, cpu_count=4)

    rows, cols, values = [], [], []
    for j, (i, v) in enumerate(output_col_data):
        rows.append(i)
        cols.append(np.ones_like(i)*j)
        values.append(v)
    rows, cols, values = [np.concatenate(x) for x in [rows, cols, values]]
    A_low_res = csc_matrix((values, (rows, cols)), shape=(low_res**2, n_strings))
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
