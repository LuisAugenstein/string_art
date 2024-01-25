from scipy.sparse import csr_matrix
import numpy as np


def multi_sample_correspondence_map(low_res: int, high_res: int) -> csr_matrix:
    """
    Returns
    -
    correspondence_map: np.shape([low_res**2, high_res**2])
    """
    super_sampling_factor = high_res // low_res
    n_pixels = high_res**2
    n_correspondence_values = low_res**2

    row_ind = np.arange(n_correspondence_values).reshape(low_res, low_res)
    row_ind = row_ind.repeat(super_sampling_factor, axis=0).repeat(super_sampling_factor, axis=1).flatten()
    col_ind = np.arange(n_pixels)
    v = np.ones(n_pixels) / (super_sampling_factor ** 2)
    return csr_matrix((v, (row_ind, col_ind)), shape=(n_correspondence_values, n_pixels))
