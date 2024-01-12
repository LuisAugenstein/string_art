import os
from scipy.sparse import load_npz, save_npz, csr_matrix
from scipy.io import loadmat
from string_art.preprocessing import precompute_string_matrices
from string_art.io.root_path import root_path
import numpy as np


def load_string_matrices(n_pins: int, pin_side_length: float, string_thickness: float, min_angle: float, high_res: int, low_res: float) -> tuple[csr_matrix, csr_matrix, np.ndarray]:
    """
    Returns
    -
    A_high_res: np.shape([high_res**2, n_edges], dtype=int)  binary matrix where 1 at (i,j) means that drawing edge j will cross pixel i  
    A_low_res: np.shape([low_res**2, n_edges], dtype=float)  resized A_high_res with values between 0 and 1
    valid_edges_mask: np.shape([n_edges], dtype=bool)        False for excluding edges from the optimization.
    """
    string_matrices_dir = f"{root_path}/data/string_matrices"
    config_dir = f'{string_matrices_dir}/{n_pins}_{pin_side_length}_{string_thickness}_{min_angle}_{high_res}_{low_res}'
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    matrix_paths = [f'{config_dir}/A_high_res.mat', f'{config_dir}/A_low_res.mat', f'{config_dir}/valid_edges_mask.npy']
    if os.path.exists(config_dir) and all([os.path.exists(path) for path in matrix_paths]):
        # high_res_matrix, low_res_matrix = [load_npz(path) for path in matrix_paths[:2]]
        A_high_res, A_low_res = [loadmat(path)['A'] for path in matrix_paths[:2]]
        valid_edges_mask = np.load(matrix_paths[2])
    else:
        A_high_res, A_low_res, valid_edges_mask = precompute_string_matrices(
            n_pins, pin_side_length, string_thickness, min_angle, high_res, low_res)
        save_npz(matrix_paths[0], A_high_res)
        save_npz(matrix_paths[1], A_low_res)
        np.save(matrix_paths[2], valid_edges_mask)
    return A_high_res, A_low_res, valid_edges_mask
