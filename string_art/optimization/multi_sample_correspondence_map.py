import numpy as np
from scipy.sparse import csr_matrix
from skimage.transform import resize


def multi_sample_correspondence_map(low_res: int, super_sampling_factor: int):
    high_res = low_res * super_sampling_factor
    n_pixels = high_res**2
    n_correspondence_values = low_res**2

    row_ind = np.reshape(np.arange(n_correspondence_values), (low_res, low_res)).T
    row_ind = resize(row_ind, (high_res, high_res), mode='constant').T.flatten()
    col_ind = np.arange(n_pixels)
    v = np.ones(n_pixels) / (super_sampling_factor ** 2)
    corresponence_map = csr_matrix((v, (row_ind, col_ind)), shape=(n_correspondence_values, n_pixels))
    return corresponence_map
