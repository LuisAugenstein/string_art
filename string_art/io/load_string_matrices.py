import os
from scipy.sparse import load_npz, save_npz, csr_matrix
from scipy.io import loadmat
from string_art.preprocessing import precompute_string_matrices
from string_art.io.root_path import root_path
import numpy as np


def load_string_matrices(n_pins: int, pin_side_length: float, string_thickness: float, min_angle: float, high_res: int, low_res: float) -> tuple[csr_matrix, csr_matrix, np.ndarray]:
    string_matrices_dir = f"{root_path}/data/string_matrices"
    config_dir = f'{string_matrices_dir}/{n_pins}_{pin_side_length}_{string_thickness}_{min_angle}_{high_res}_{low_res}'
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    matrix_paths = [f'{config_dir}/A_high_res.mat', f'{config_dir}/A_low_res.mat', f'{config_dir}/fabricable.npy']
    if os.path.exists(config_dir) and all([os.path.exists(path) for path in matrix_paths]):
        # high_res_matrix, low_res_matrix = [load_npz(path) for path in matrix_paths[:2]]
        high_res_matrix, low_res_matrix = [loadmat(path)['A'] for path in matrix_paths[:2]]
        fabricable_edges = np.load(matrix_paths[2])
    else:
        high_res_matrix, low_res_matrix, fabricable_edges = precompute_string_matrices(
            n_pins, pin_side_length, string_thickness, min_angle, high_res, low_res)
        save_npz(matrix_paths[0], high_res_matrix)
        save_npz(matrix_paths[1], low_res_matrix)
        np.save(matrix_paths[2], fabricable_edges)
    return high_res_matrix, low_res_matrix, fabricable_edges
