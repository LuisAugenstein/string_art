import numpy as np
import cupy
import cupyx.scipy as cipy
import scipy


def get_np_array_module(a: np.ndarray | cupy.ndarray):
    """
    Returns
    xp: either numpy or cupy
    xipy: either scipy or cupyx
    """
    if cupy.get_array_module(a) == cupy:
        return cupy, cipy
    return np, scipy


def get_np_array_module_bool(cuda: bool):
    return (cupy, cipy) if cuda else (np, scipy)
