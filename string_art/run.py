import numpy as np
from string_art.config import Config
from PIL import Image


def run(image: np.ndarray, config: Config):
    """
    image: np.shape([N, N])  square greyscale image with values between 0 and 255
    """
    image = np.array(Image.fromarray(image).resize((config.low_resolution, config.low_resolution)))
    image = (image - np.min(image)) // np.max(image)
