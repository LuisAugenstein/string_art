import numpy as np
from string_art.config import Config
from PIL import Image


def create_circular_mask(size: int, radius: float = None):
    y, x = np.ogrid[:size, :size]
    center = (size-1) // 2
    if radius is None:
        radius = center
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    return mask


def resize_image(image: np.ndarray, resolution: int) -> np.ndarray:
    """
    Parameters
    -
    image: np.shape([N, N], dtype=np.uint8) square grayscale image with values between 0 and 255

    Returns
    -
    np.shape([resolution, resolution], dtype=np.float32) maximum contrast grayscale image with values between 0 and 1
    """
    image = np.array(Image.fromarray(image).resize((resolution, resolution)), dtype=np.float32)
    image = (image - np.min(image)) / np.max(image)
    return image


def mask_image(image: np.ndarray) -> np.ndarray:
    """
    Parameters
    -
    image: np.shape([N, N], dtype=np.float32) square grayscale image with values between 0 and 1

    Returns
    -
    masked_image: np.shape([N, N], dtype=np.float32) pixels outside the circular frame are set to 1 (white)
    mask: np.shape([low_resolution, low_resolution], dtype=np.bool) binary mask which is 1 for pixels inside the circular frame
    """
    mask = create_circular_mask(image.shape[0])
    image[~mask] = 1.0
    return image, mask
