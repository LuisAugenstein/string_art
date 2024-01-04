from scipy.sparse import csr_matrix, find
import numpy as np


def split_apart_string_matrix(A: csr_matrix):
    A_edge_indices_to_pixel_codes = []
    _, n_edges = A.shape
    for k in range(n_edges):
        indices, _, val = find(A[:, k])
        A_edge_indices_to_pixel_codes.append((np.uint32(indices), val))

    EPI, EPV = np.hstack(A_edge_indices_to_pixel_codes)
    EPI = EPI.astype(np.uint32)
    CEI = np.uint32(np.arange(n_edges).repeat([len(a) for a, _ in A_edge_indices_to_pixel_codes]))
    m = EPI.shape[0]
    n = int(np.max(EPI)+1)

    row_indices = np.arange(m)
    col_indices = np.array(EPI)
    data = np.ones_like(row_indices)

    ITI = csr_matrix((data, (row_indices, col_indices)), shape=(m, n))
    # edge_pixel_indices, edge_pixel_values, correspondence_edge_indices, ???
    return EPI, EPV, CEI, ITI
