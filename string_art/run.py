import os
import numpy as np
from string_art.config import Config
import imageio.v3 as imageio
from string_art.preprocessing.preprocess_image import preprocess_image
from string_art.preprocessing import get_pins, precompute_string_matrices
from string_art.io import load_picked_edges, load_string_matrices, load_error_image, root_path
from skimage.transform import resize


def run(image: np.ndarray, config: Config, name_of_the_run: str):
    """
    image: np.shape([N, N])  square greyscale image with values between 0 and 255
    """
    image, mask = preprocess_image(image, config)

    # least squares solution
    # mask = mask.reshape(-1)
    # y = image.reshape(-1)[mask]
    # m = y.size()[0]
    # n = 4 * comb(config.n_pins, 2)
    # x = np.linalg.lstsq(A, y)

    A_high_res, A_low_res, fabricable = load_string_matrices(config.n_pins, config.pin_side_length, config.string_thickness,
                                                             config.min_angle, config.high_resolution, config.low_resolution)
    # x, pickedEdgesSequence = optimize_strings_greedy_multi_sampling(
    #     image, img, lowRes, superSamplingWindowWidth, minAngle, numPins, importanceMap, matrixPath)

    # visualize string path instead of writing the original image
    # image = (image*255).astype(np.uint8)
    # imageio.imwrite(f'{root_path}/data/outputs/output.png', image)

    x, picked_edges_sequence = load_picked_edges(image, config.super_sampling_window_width,
                                                 config.min_angle, config.n_pins, A_high_res, A_low_res, fabricable)

    recon = np.minimum(1, np.dot(A_high_res, x))
    recon_image_high = np.reshape(recon, (A_high_res, config.high_resolution))
    recon_image_high = np.flipud(recon_image_high.T)
    recon_image_low = resize(recon_image_high, (config.low_resolution, config.low_resolution), mode='constant')

    if config.invert_output:
        recon_image_high = 1 - recon_image_high
        recon_image_low = 1 - recon_image_low

    rmse_value = np.sqrt(np.mean((image - recon_image_low[mask])**2))
    print('RMSE: ', rmse_value)

    load_error_image(name_of_the_run, image, recon_image_low)
    # load_consecutive_path(name_of_the_run, x, config.n_pins, config.low_resolution,)
