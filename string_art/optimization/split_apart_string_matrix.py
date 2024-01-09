from scipy.sparse import csr_matrix, find
import numpy as np


def get_index_to_index_map(A: csr_matrix) -> csr_matrix:
    """
    Parameters
    -
    A: np.shape([n_pixels, n_edges])    values between 0 and 1 indicate how much a pixel i is darkened if edge j is active.

    Returns
    -
    index_to_index_map: np.shape([n_values_in_A, n_pixels]) binary matrix which contains a single 1 in each row (i,edge_pixel_indices[i]) and otherwise 0. 
    """
    _, edge_pixel_indices, _ = find(A.T)
    n_values_in_A = edge_pixel_indices.shape[0]
    data = np.ones(n_values_in_A)
    index_to_index_map = csr_matrix((data, (np.arange(n_values_in_A), edge_pixel_indices)), shape=(n_values_in_A, A.shape[0]))
    return index_to_index_map
