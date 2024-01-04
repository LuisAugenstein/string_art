import numpy as np
from string_art.evaluations.color_gradient import ColorGradient


def get_error_image(target_image: np.ndarray, recon_image_low: np.ndarray):
    gradient = ColorGradient(0.0, 1.0)
    error_image = np.abs(target_image - recon_image_low)
    heat_map = gradient.color_at_value(error_image)
    return heat_map
