import os
import numpy as np
from string_art.config import Config
import imageio.v3 as imageio
from string_art.preprocessing.preprocess_image import preprocess_image
from string_art.preprocessing import get_pins, precompute_string_matrices
from string_art.io import load_string_matrices, root_path


def run(image: np.ndarray, config: Config):
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

    A_high_res, A_low_res = load_string_matrices(config.n_pins, config.pin_side_length,
                                                 config.string_thickness, config.high_resolution, config.low_resolution)
    # x, pickedEdgesSequence = optimize_strings_greedy_multi_sampling(
    #     image, img, lowRes, superSamplingWindowWidth, minAngle, numPins, importanceMap, matrixPath)

    # visualize string path instead of writing the original image
    image = (image*255).astype(np.uint8)
    imageio.imwrite(f'{root_path}/data/outputs/output.png', image)
