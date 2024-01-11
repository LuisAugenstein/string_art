import numpy as np
from matplotlib import cm


def get_error_image(target_image: np.ndarray, recon_image_low: np.ndarray):
    error_image = np.abs(target_image - recon_image_low)
    heat_map = cm.viridis(error_image)
    return heat_map
