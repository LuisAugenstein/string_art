import numpy as np
from typing import Literal
import torch


def indices_1D_high_res_to_low_res(high_res_indices: np.ndarray, high_res: int, low_res: int) -> np.ndarray:
    """
    Parameters
    -
    high_res_indices: np.shape([n_values])  values between 0 and high_res**2 representing a 1D pixel location

    Returns
    -
    low_res_indices: np.shape([n_values])   values between 0 and low_res**2 representing the corresponding 1D pixel location in the low_res image
    """
    super_sampling_factor = high_res // low_res
    rows = (high_res_indices // high_res) // super_sampling_factor
    cols = (high_res_indices % high_res) // super_sampling_factor
    low_res_indices = rows * low_res + cols
    return low_res_indices


def indices_1D_low_res_to_high_res(low_res_indices: np.ndarray, low_res: int, high_res: int) -> np.ndarray:
    points_low_res = indices_1D_to_2D(low_res_indices, low_res, 'row-col')
    super_sampling_factor = high_res // low_res
    offsets = np.vstack([np.array([i, j]) for i in range(super_sampling_factor) for j in range(super_sampling_factor)])
    points_low_res = points_low_res[:, None, :]*super_sampling_factor + offsets[None, :, :]
    high_res_indices = indices_2D_to_1D(points_low_res[:, :, 1], points_low_res[:, :, 0], high_res)
    return high_res_indices


def indices_2D_to_1D(x: np.ndarray, y: np.ndarray, domain_width: float) -> np.ndarray:
    return y * domain_width + x


def indices_1D_to_2D(i: torch.Tensor, domain_width: float, mode: Literal['x-y', 'row-col'] = 'x-y') -> torch.Tensor:
    if not torch.is_tensor(i):
        return np.vstack([i % domain_width, i // domain_width]).T if mode == 'x-y' else np.vstack([i // domain_width, i % domain_width]).T

    if mode == 'x-y':
        return torch.vstack([i % domain_width, i // domain_width]).T
    return torch.vstack([i // domain_width, i % domain_width]).T
