from string_art.io import load_string_matrices
from string_art.config import get_default_config
from scipy.io import loadmat
from scipy.sparse import find, csr_matrix
import numpy as np
from string_art.preprocessing import high_res_to_low_res_indices, high_res_to_low_res_indices_optimized
from tests.utils import measure_time


def test_with_matlab_string_matrices():
    """
    Check whether the high resolution string matrix is computed equally than in the matlab code.
    """
    config = get_default_config()
    config.n_pins = 16
    A_high_res, _, _ = load_string_matrices(config.n_pins, config.pin_side_length,
                                            config.string_thickness, config.min_angle, config.high_res, config.low_res)
    A_high_res_matlab = loadmat(f'tests/data/A_high_res.mat')['A']
    assert sparse_matrices_all_close(A_high_res, A_high_res_matlab, first_n_edges=50)


def test_string_matrix_resizing():
    A_high_res, A_low_res_matlab = [loadmat(f'tests/data/{file_name}')['A'] for file_name in ['A_high_res.mat', 'A_low_res.mat']]
    low_res = int(np.sqrt(A_low_res_matlab.shape[0]))

    col_index = 0  # arbitrary column index
    i_high_res, _, v_high_res = find(A_high_res[:, col_index])
    (i_new, v_new), t = measure_time(lambda: high_res_to_low_res_indices(i_high_res, v_high_res, A_high_res.shape[0], low_res))
    (i_new2, v_new2), t2 = measure_time(lambda: high_res_to_low_res_indices_optimized(i_high_res, v_high_res, A_high_res.shape[0], low_res))
    i_matlab, _, v_matlab = find(A_low_res_matlab[:, col_index])
    sort_indices = np.argsort(i_new)
    assert np.allclose(i_new[sort_indices], i_matlab) and np.allclose(v_new[sort_indices], v_matlab)
    assert np.allclose(i_new2[sort_indices], i_matlab) and np.allclose(v_new2[sort_indices], v_matlab)


def sparse_matrices_all_close(A: csr_matrix, B: csr_matrix, first_n_edges=None) -> bool:
    if first_n_edges is None:
        first_n_edges = A.shape[1]
    for edge_index in range(first_n_edges):
        i, j, v = find(A[:, edge_index])
        i2, j2, v2 = find(B[:, edge_index])
        if not np.allclose(i, i2) or not np.allclose(j, j2) or not np.allclose(v, v2):
            return False
    return True
