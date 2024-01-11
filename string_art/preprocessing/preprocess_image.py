import numpy as np
from string_art.transformations import imresize


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
    image = imresize(image, output_shape=(resolution, resolution))
    image = np.clip(image, 0, 1)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    if invert:
        image = 1 - image
    return np.flipud(image)
