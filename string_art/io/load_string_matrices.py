import os
from scipy.sparse import load_npz, save_npz, csc_matrix
from string_art.preprocessing import precompute_string_matrix, high_res_to_low_res_matrix
from string_art.io.root_path import root_path
from string_art.io.mkdir import mkdir
import numpy as np


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

    high_res_path, valid_edges_mask_path = f'{config_dir}/A_high_res.npz', f'{config_dir}/valid_edges_mask.npy'
    if os.path.exists(high_res_path) and os.path.exists:
        A_high_res, valid_edges_mask = load_npz(high_res_path), np.load(valid_edges_mask_path)
    else:
        A_high_res, valid_edges_mask = precompute_string_matrix(n_pins, pin_side_length, string_thickness, min_angle, high_res)
        mkdir(config_dir)
        save_npz(high_res_path, A_high_res)
        np.save(valid_edges_mask_path, valid_edges_mask)

    low_res_path = f'{config_dir}/A_low_res.npz'
    if os.path.exists(low_res_path):
        A_low_res = load_npz(low_res_path)
    else:
        A_low_res = high_res_to_low_res_matrix(A_high_res, low_res)
        save_npz(low_res_path, A_low_res)

    return A_high_res, A_low_res, valid_edges_mask
