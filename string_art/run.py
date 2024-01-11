import numpy as np
from string_art.config import Config
import imageio.v3 as imageio
from string_art.preprocessing.preprocess_image import mask_image, resize_image
from string_art.io import load_picked_edges, load_string_matrices, load_error_image, root_path
from skimage.transform import resize
from scipy.io import loadmat
import matplotlib.pyplot as plt


def run(image: np.ndarray, config: Config, name_of_the_run: str):
    """
    image: np.shape([N, N])  square greyscale image with values between 0 and 255
    """
    image = resize_image(image, config.low_resolution)
    image = loadmat(f'{root_path}/data/inputs/preprocessed_cat_img.mat')['img']
    masked_image, mask = mask_image(image.copy())
    if config.invert_input:
        masked_image = 1 - masked_image
        # image = 1 - image
    imageio.imwrite(f'{root_path}/data/outputs/masked_image.png', (masked_image*255).astype(np.uint8))

    # Precompute string matrices
    A_high_res, A_low_res, valid_edges_mask = load_string_matrices(config.n_pins, config.pin_side_length, config.string_thickness,
                                                                   config.min_angle, config.high_resolution, config.low_resolution)

    # find/load optimal edges to approximate the image
    importance_map = np.ones((config.low_resolution, config.low_resolution))
    importance_map[~mask] = 0
    x = load_picked_edges(name_of_the_run, image, importance_map, A_high_res, A_low_res, valid_edges_mask)

    # reconstruct image
    recon = np.clip(A_high_res @ x, 0, 1)
    recon_image_high = recon.reshape(config.high_resolution, config.high_resolution)
    recon_image_high = np.flipud(recon_image_high.T)
    recon_image_low = resize(recon_image_high, (config.low_resolution, config.low_resolution), mode='constant')

    if config.invert_output:
        recon_image_high = 1 - recon_image_high
        recon_image_low = 1 - recon_image_low

    rmse_value = np.sqrt(np.mean((image[mask] - recon_image_low[mask])**2))
    print('RMSE: ', rmse_value)

    load_error_image(name_of_the_run, image, recon_image_low)
    # load_consecutive_path(name_of_the_run, x, config.n_pins, config.low_resolution,)
