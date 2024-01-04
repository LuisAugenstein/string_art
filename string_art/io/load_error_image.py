import numpy as np
from string_art.evaluations import get_error_image
import imageio
from string_art.io.root_path import root_path
import os


def load_error_image(name_of_the_run: str, target_image: np.ndarray, recon_image_low: np.ndarray):
    project_dir = f'{root_path}/data/outputs/{name_of_the_run}'
    if not os.path.exists(project_dir):
        os.mkdir(project_dir)
    image_path = f'{project_dir}/error_image.png'
    if os.path.exists(image_path):
        return imageio.imread(image_path)
    error_image = get_error_image(target_image, recon_image_low)
    imageio.imwrite(image_path, error_image)
    return error_image
