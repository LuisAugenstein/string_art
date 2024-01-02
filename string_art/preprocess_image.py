import numpy as np
from string_art.config import Config
from PIL import Image


def create_circular_mask(size):
    y, x = np.ogrid[:size, :size]
    center = (size-1) // 2
    radius = center
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    return mask


def preprocess_image(image: np.ndarray, config: Config) -> np.ndarray:
    """
    Parameters
    -
    image: np.shape([N, N], dtype=np.uint8) square greyscale image with values between 0 and 255

    Returns
    -
    preprocessed_image: np.shape([low_resolution, low_resolution], dtype=np.float32) high contrast greyscale image with values between 0 and 1
    mask: np.shape([low_resolution, low_resolution], dtype=np.bool) binary mask which is 1 for pixels inside the circular frame
    """
    image = np.array(Image.fromarray(image).resize((config.low_resolution, config.low_resolution)), dtype=np.float32)
    image = (image - np.min(image)) / np.max(image)
    mask = create_circular_mask(config.low_resolution)
    image[~mask] = 1.0
    return image, mask
