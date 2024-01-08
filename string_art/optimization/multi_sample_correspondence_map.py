import numpy as np
from scipy.sparse import csr_matrix
from skimage.transform import resize


def multi_sample_correspondence_map(low_res: int, high_res: int):
    super_sampling_factor = high_res // low_res
    n_pixels = high_res**2
    n_correspondence_values = low_res**2

    row_ind = np.arange(n_correspondence_values).reshape(low_res, low_res)
    row_ind = row_ind.repeat(super_sampling_factor, axis=0).repeat(super_sampling_factor, axis=1).flatten()
    col_ind = np.arange(n_pixels)
    v = np.ones(n_pixels) / (super_sampling_factor ** 2)
    corresponence_map = csr_matrix((v, (row_ind, col_ind)), shape=(n_correspondence_values, n_pixels))
    return corresponence_map
