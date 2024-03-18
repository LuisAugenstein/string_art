import torch


def find(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    reimplements the scipy find function for a sparse matrix in torch

    Parameters
    -
    A: torch.shape([N, M])   sparse_coo matrix with n_nonzero elements

    Returns
    -
    cols: torch.shape([n_nonzero], dtype=int)  column indices of the non-zero elements
    rows: torch.shape([n_nonzero], dtype=int)  row indices of the non-zero elements
    values: torch.shape([n_nonzero], dtype=float)  values of the non-zero elements
    """
    A = A.to_sparse_coo()
    cols, sorted_indices = A.indices()[1].sort()
    rows = A.indices()[0, sorted_indices]
    values = A.values()[sorted_indices]
    return cols, rows, values
