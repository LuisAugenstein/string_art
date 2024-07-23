import numpy as np


def create_circular_mask(image_size: int, epsilon: float = 0.01) -> np.ndarray:
    """
    Returns
    circular_mask: [image_size, image_size] boolean mask which is True outside the circle and False inside
    """
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    X, Y = np.meshgrid(x, y)
    return (X ** 2 + Y ** 2) + epsilon > 1
