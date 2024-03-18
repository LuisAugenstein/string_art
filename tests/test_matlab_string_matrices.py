import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from string_art.config import get_config
from string_art.preprocessing import precompute_string_matrix, high_res_to_low_res_string_matrix, high_res_to_low_res_string_indices, high_res_to_low_res_string_indices_optimized
from tests.utils import measure_time
from typing import Callable
from tqdm import tqdm


def are_string_matrices_equal(A: csc_matrix, B: csc_matrix) -> bool:
    for j in range(A.shape[1]):
        if not np.allclose(A[:, j].indices, B[:, j].indices) or not np.allclose(A[:, j].data, B[:, j].data):
            return False
    return True


def test_with_matlab_string_matrices():
    """
    Check whether the high resolution string matrix is computed equally than in the matlab code.
    """
    def compare_high_res_string_matrices(n_pins: int) -> bool:
        config = get_config()
        A_high_res, _ = precompute_string_matrix(n_pins, config.pin_side_length, config.string_thickness, config.min_angle, config.high_res)
        A_high_res_matlab = loadmat(f'tests/data/A_high_res_{n_pins}_pins.mat')['A']
        return are_string_matrices_equal(A_high_res, A_high_res_matlab)

    assert compare_high_res_string_matrices(n_pins=16)
    assert compare_high_res_string_matrices(n_pins=32)


def test_string_matrix_resizing():
    def compare_low_res_string_matrices(index_transformation_function: Callable, n_pins: int, ) -> bool:
        A_high_res, A_low_res_matlab = [loadmat(f'tests/data/{file_name}')['A']
                                        for file_name in [f'A_high_res_{n_pins}_pins.mat', f'A_low_res_{n_pins}_pins.mat']]
        high_res, low_res = int(np.sqrt(A_high_res.shape[0])), int(np.sqrt(A_low_res_matlab.shape[0]))
        for j in tqdm(range(A_high_res.shape[1])):
            i_true, v_true = A_low_res_matlab[:, j].indices, A_low_res_matlab[:, j].data
            i, v = index_transformation_function(A_high_res[:, j].indices, A_high_res[:, j].data, high_res, low_res)
            sorted_indices = np.argsort(i)
            i, v = i[sorted_indices], v[sorted_indices]
            if not np.allclose(i, i_true) or not np.allclose(v, v_true):
                return False
        return True

    assert compare_low_res_string_matrices(high_res_to_low_res_string_indices, n_pins=16)
    assert compare_low_res_string_matrices(high_res_to_low_res_string_indices_optimized, n_pins=16)
    assert compare_low_res_string_matrices(high_res_to_low_res_string_indices_optimized, n_pins=32)


def test_high_res_to_low_res_indices_optimized():
    """
    Check that the optimized resizing function is up to 15 times faster than the original matlab implementation,
    especially for short strings.
    """
    A_high_res = loadmat('tests/data/A_high_res_16_pins.mat')['A']
    high_res = int(np.sqrt(A_high_res.shape[0]))
    low_res = high_res // 8
    column_index = 0  # arbitrary value
    (i, v), t = measure_time(lambda: high_res_to_low_res_string_indices(
        A_high_res[:, column_index].indices, A_high_res[:, column_index].data, high_res, low_res))
    (i2, v2), t2 = measure_time(lambda: high_res_to_low_res_string_indices_optimized(
        A_high_res[:, column_index].indices, A_high_res[:, column_index].data, high_res, low_res))
    assert np.allclose(i, i2) and np.allclose(v, v2)
    speedup = t // t2
    assert speedup > 15
