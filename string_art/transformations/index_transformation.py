import numpy as np
from typing import Literal


def indices_1D_high_res_to_low_res(high_res_indices: np.ndarray, high_res: int, low_res: int) -> np.ndarray:
    row_high_res, col_high_res = indices_1D_to_2D(high_res_indices, high_res, 'row-col').T
    super_sampling_factor = high_res // low_res
    row, col = row_high_res // super_sampling_factor, col_high_res // super_sampling_factor
    low_res_index = indices_2D_to_1D(col, row, low_res)
    return low_res_index


def indices_1D_low_res_to_high_res(low_res_indices: np.ndarray, low_res: int, high_res: int) -> np.ndarray:
    points_low_res = indices_1D_to_2D(low_res_indices, low_res, 'row-col')
    super_sampling_factor = high_res // low_res
    offsets = np.vstack([np.array([i, j]) for i in range(super_sampling_factor) for j in range(super_sampling_factor)])
    points_low_res = points_low_res[:, None, :]*super_sampling_factor + offsets[None, :, :]
    high_res_indices = indices_2D_to_1D(points_low_res[:, :, 1], points_low_res[:, :, 0], high_res)
    return high_res_indices


def indices_2D_to_1D(x: np.ndarray, y: np.ndarray, domain_width: float) -> np.ndarray:
    return y * domain_width + x


def indices_1D_to_2D(i: np.ndarray, domain_width: float, mode: Literal['x-y', 'row-col'] = 'x-y') -> np.ndarray:
    return np.vstack([i % domain_width, i // domain_width]).T if mode == 'x-y' else np.vstack([i // domain_width, i % domain_width]).T
