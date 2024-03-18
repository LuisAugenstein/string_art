import numpy as np
from string_art.transformations import indices_1D_to_2D
from string_art.utils import map, find
import torch


def high_res_to_low_res_string_matrix(A_high_res: torch.Tensor, low_res: int) -> torch.Tensor:
    """
    Parameters
    -
    A_high_res: torch.shape([high_res**2, n_strings])   values between 0 and 1 indicate how much a pixel i is darkened if edge j is active.
    low_res: int                                        resolution of the low_res image

    Returns
    -
    A_low_res: torch.shape([low_res**2, n_strings])      resized A_high_res with values between 0 and 1
    """

    def col_mapping(col: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        i, v = col
        if i.shape[0] == 0:
            return i, v
        high_res = int(np.sqrt(A_high_res.shape[0]))
        # return high_res_to_low_res_string_indices(i, v, high_res, low_res)
        return high_res_to_low_res_string_indices_optimized(i, v, high_res, low_res)

    print(f'Compute A_low_res for low_res={low_res}')
    n_strings = A_high_res.shape[1]

    A_high_res_csc = A_high_res.to_sparse_csc()
    ccol = A_high_res_csc.ccol_indices()
    col_data = [(A_high_res_csc.row_indices()[ccol[j]:ccol[j+1]], A_high_res_csc.values()[ccol[j]:ccol[j+1]]) for j in range(n_strings)]

    output_col_data = map(col_mapping, col_data)

    rows, cols, values = [], [], []
    for j, (i, v) in enumerate(output_col_data):
        rows.append(i)
        cols.append(torch.ones_like(i)*j)
        values.append(v)
    rows, cols, values = [torch.concatenate(x) for x in [rows, cols, values]]
    indices = torch.stack([rows, cols])
    A_low_res = torch.sparse_coo_tensor(indices, values, size=(low_res**2, n_strings))
    print(f'A_low_res.shape={A_low_res.shape[0]}x{A_low_res.shape[1]}')
    return A_low_res


def high_res_to_low_res_string_indices(high_res_indices: torch.Tensor, high_res_values: torch.Tensor, high_res: int, low_res: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    -
    high_res_indices: torch.shape([n_string_pixels_high_res])    flattened indices of the high_res string
    high_res_values: torch.shape([n_string_pixels_high_res])     values of the high_res string

    Returns
    -
    low_res_indices: torch.shape([n_string_pixels_low_res])      flattened indices of the low_res string
    low_res_values: torch.shape([n_string_pixels_low_res])       values of the low_res string (mean of corresponding high res pixel values)
    """
    scale = high_res//low_res
    img = torch.zeros(high_res**2)
    img[high_res_indices] = high_res_values
    low_res_img = img.reshape((low_res, scale, low_res, scale)).mean(dim=(1, 3))
    k, j, v = find(low_res_img)
    i = j * low_res + k
    return i, v


def high_res_to_low_res_indices(high_res_indices: torch.Tensor, high_res_values: torch.Tensor, high_res: int, low_res: int) -> tuple[torch.Tensor, torch.Tensor]:
    scale = high_res//low_res
    img = torch.zeros(high_res**2)
    img[high_res_indices] = high_res_values
    low_res_img = img.reshape((low_res, scale, low_res, scale)).mean(dim=(1, 3))
    k, j, v = find(low_res_img)
    i = j * low_res + k
    return i, v


def high_res_to_low_res_string_indices_optimized(high_res_indices: torch.Tensor, high_res_values: torch.Tensor, high_res: int, low_res: int) -> tuple[torch.Tensor, torch.Tensor]:
    scale = high_res//low_res
    xy = indices_1D_to_2D(high_res_indices, high_res, mode='row-col')  # [N,2]
    bbox_start, _ = torch.min(xy, dim=0)
    bbox_end, _ = torch.max(xy, dim=0)
    bbox_start -= bbox_start % scale
    bbox_end += scale - bbox_end % scale

    x, y = xy.T - bbox_start[:, None]
    bbox_end -= bbox_start
    img = torch.zeros((bbox_end[0], bbox_end[1]))
    img[x, y] = high_res_values
    low_res_img = img.reshape((bbox_end[0]//scale, scale, bbox_end[1]//scale, scale)).mean(dim=(1, 3))
    k, j, v = find(low_res_img)
    j = j + bbox_start[0]//scale
    k = k + bbox_start[1]//scale
    i = j * low_res + k
    return i, v
