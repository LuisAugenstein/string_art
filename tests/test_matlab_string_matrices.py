import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from string_art.config import get_config
from string_art.optimization.losses.high_res_to_low_res_matrix import high_res_to_low_res_matrix
from string_art.preprocessing import precompute_string_matrix
import torch


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


def test_string_matrix_resizing2():
    torch.set_default_dtype(torch.float64)

    def compare_low_res_string_matrices(n_pins: int) -> bool:
        A_low_res_matlab: csc_matrix = loadmat(f'tests/data/A_low_res_{n_pins}_pins.mat')['A']
        A_high_res_matlab: csc_matrix = loadmat(f'tests/data/A_high_res_{n_pins}_pins.mat')['A']
        low_res, high_res = int(np.sqrt(A_low_res_matlab.shape[0])), int(np.sqrt(A_high_res_matlab.shape[0]))
        h2l = high_res_to_low_res_matrix(low_res, high_res)
        h2l_np = csc_matrix((h2l.values(), (h2l.indices()[0], h2l.indices()[1])), shape=h2l.shape)
        A_low_res_hand = h2l_np @ A_high_res_matlab
        for j in range(A_low_res_hand.shape[1]):
            i_true, v_true = A_low_res_matlab[:, j].indices, A_low_res_matlab[:, j].data
            i_hand, v_hand = A_low_res_hand[:, j].indices, A_low_res_hand[:, j].data
            sorted_indices = np.argsort(i_hand)
            i, v = i_hand[sorted_indices], v_hand[sorted_indices]
            if not np.allclose(i, i_true) or not np.allclose(v, v_true):
                return False
        return True
    assert compare_low_res_string_matrices(n_pins=16)
    assert compare_low_res_string_matrices(n_pins=32)
