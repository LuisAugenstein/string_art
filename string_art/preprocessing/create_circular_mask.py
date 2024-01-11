import numpy as np
from PIL import Image


def create_circular_mask(size: int, radius: float = None) -> np.ndarray:
    """
    Returns
    -
    mask: np.shape([size, size], dtype=np.bool) mask with True values inside the circle and False values outside the circle
    """
    y, x = np.ogrid[:size, :size]
    center = (size-1) // 2
    if radius is None:
        radius = center
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    return mask
