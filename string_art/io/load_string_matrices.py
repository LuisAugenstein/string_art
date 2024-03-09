import os
from scipy.sparse import load_npz, save_npz, csc_matrix
from string_art.preprocessing import precompute_string_matrix, high_res_to_low_res_matrix
from string_art.io.root_path import root_path
from string_art.io.mkdir import mkdir
import numpy as np
import torch


def load_string_matrices(n_pins: int, pin_side_length: float, string_thickness: float, min_angle: float, high_res: int, low_res: float) -> tuple[csc_matrix, csc_matrix, np.ndarray]:
    """
    Returns
    -
    A_high_res: np.shape([high_res**2, n_strings], dtype=int)  binary matrix where 1 at (i,j) means that drawing edge j will cross pixel i  
    A_low_res: np.shape([low_res**2, n_strings], dtype=float)  resized A_high_res with values between 0 and 1
    valid_edges_mask: np.shape([n_strings], dtype=bool)        False for excluding edges from the optimization.
    """
    string_matrices_dir = f"{root_path}/data/string_matrices"
    config_dir = f'{string_matrices_dir}/{n_pins}_{pin_side_length}_{string_thickness}_{min_angle:.4f}_{high_res}_{low_res}'

    high_res_path, valid_edges_mask_path = f'{config_dir}/A_high_res.pt', f'{config_dir}/valid_edges_mask.pt'
    if os.path.exists(high_res_path) and os.path.exists:
        A_high_res, valid_edges_mask = torch.load(high_res_path), torch.load(valid_edges_mask_path)
    else:
        A_high_res, valid_edges_mask = precompute_string_matrix(n_pins, pin_side_length, string_thickness, min_angle, high_res)
        print('saving A_high_res to disk...')
        mkdir(config_dir)
        torch.save(A_high_res, high_res_path)
        torch.save(valid_edges_mask, valid_edges_mask_path)

    A_high_res = csc_matrix((A_high_res.values(), (A_high_res.indices()[0], A_high_res.indices()[1])), shape=A_high_res.shape)
    valid_edges_mask = valid_edges_mask.numpy()

    low_res_path = f'{config_dir}/A_low_res.npz'
    if os.path.exists(low_res_path):
        A_low_res = load_npz(low_res_path)
    else:
        A_low_res = high_res_to_low_res_matrix(A_high_res, low_res)
        print('saving A_low_res to disk...')
        save_npz(low_res_path, A_low_res)

    return A_high_res, A_low_res, valid_edges_mask
