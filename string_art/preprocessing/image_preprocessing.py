import numpy as np
from string_art.transformations import matlab_imresize


def preprocess_image(image: np.ndarray, resolution: int, invert: bool) -> np.ndarray:
    """
    Parameters
    -
    image: np.shape([N, N])  square input image with grayscale values between 0 and 255
    resolution: int          resolution of the output image
    invert: bool             whether to invert the image or not. Choos such that the background is light and the content dark.

    Returns
    -
    image: np.shape([resolution, resolution]) resized, normalized and (inverted) image with values between 0 and 1
    """
    image = image / 255
    image = matlab_imresize(image, output_shape=(resolution, resolution))
    image = np.clip(image, 0, 1)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    if invert:
        image = 1 - image
    return np.flipud(image)


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
