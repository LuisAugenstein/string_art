import os
from scipy.sparse import load_npz, save_npz, csr_matrix
from string_art.preprocessing import precompute_string_matrices
from string_art.io.root_path import root_path


def load_string_matrices(n_pins: int, pin_side_length: float, string_thickness: float, min_angle: float, high_res: int, low_res: float) -> tuple[csr_matrix, csr_matrix]:
    string_matrices_dir = f"{root_path}/data/string_matrices"
    config_dir = f'{string_matrices_dir}/{n_pins}_{pin_side_length}_{string_thickness}_{min_angle}_{high_res}_{low_res}'
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    matrix_paths = [f'{config_dir}/high_res_matrix.npz', f'{config_dir}/low_res_matrix.npz']
    if os.path.exists(config_dir) and all([os.path.exists(path) for path in matrix_paths]):
        high_res_matrix, low_res_matrix = [load_npz(path) for path in matrix_paths]
    else:
        high_res_matrix, low_res_matrix = precompute_string_matrices(
            n_pins, pin_side_length, string_thickness, min_angle, high_res, low_res)
        save_npz(f'{config_dir}/high_res_matrix.npz', high_res_matrix)
        save_npz(f'{config_dir}/low_res_matrix.npz', low_res_matrix)
    return high_res_matrix, low_res_matrix
