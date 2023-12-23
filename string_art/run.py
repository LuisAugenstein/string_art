import numpy as np
from string_art.config import Config
import imageio.v3 as imageio
from string_art.preprocess_image import preprocess_image


def run(image: np.ndarray, config: Config):
    """
    image: np.shape([N, N])  square greyscale image with values between 0 and 255
    """
    image, mask = preprocess_image(image, config)
    image = (image*255).astype(np.uint8)
    imageio.imwrite('data/outputs/output.png', image)
