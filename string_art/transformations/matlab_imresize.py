import numpy as np
from math import ceil

"""
implementation taken from https://github.com/fatheral/matlab_imresize
"""


def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f


def contributions(in_length, out_length, kernel=cubic, k_width=4.0):
    scale = out_length / in_length
    x = np.arange(1, out_length+1).astype(np.float64)
    u = (x + (scale - 1) / 2) / scale
    scale = np.clip(scale, -np.inf, 1.0)
    def h(x): return scale * kernel(scale * x)
    kernel_width = k_width / scale

    left = np.floor(u - kernel_width / 2)
    ind = np.expand_dims(left, axis=1) + np.arange(-1, ceil(kernel_width) + 1)  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizevec(inimg: np.ndarray, weights: np.ndarray, indices: np.ndarray, dim: int) -> np.ndarray:
    """
    inimg: np.shape([width, height])
    weights: np.shape([low_res, n_channels])
    indices: np.shape([low_res, n_channels])
    """
    if dim == 0:
        weights = np.expand_dims(weights, axis=-1)  # low_res, n_channels, 1
        img = inimg[indices]                        # low_res, n_channels, height
        outimg = np.sum(weights*img, axis=1)        # low_res, height
    elif dim == 1:
        weights = np.expand_dims(weights, axis=0)   # 1, low_res, n_channels
        img = inimg[:, indices]                     # low_res, low_res, n_channels
        outimg = np.sum(weights*img, axis=2)        # low_res, low_res

    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def imresize(I, output_shape=None):
    B = np.copy(I)
    for k in range(2):
        weights, indices = contributions(I.shape[k], output_shape[k])
        B = imresizevec(B, weights.squeeze(), indices.squeeze(), k)
    return B
