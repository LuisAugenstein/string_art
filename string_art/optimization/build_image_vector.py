import numpy as np
from scipy.sparse import csr_matrix


def build_image_vector(img: np.ndarray) -> csr_matrix:
    img_vector = img.T.flatten()
    row_ind = np.arange(img_vector.shape[0])
    col_ind = np.zeros_like(row_ind)
    return csr_matrix((img_vector, (row_ind, col_ind)), shape=(img_vector.shape[0], 1))
