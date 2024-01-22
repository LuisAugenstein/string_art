import numpy as np
import matplotlib.pyplot as plt
from string_art.io import load_picked_edges, load_string_matrices, load_error_image
from string_art.transformations import matlab_imresize
from string_art.config import Config
from string_art.preprocessing import create_circular_mask, preprocess_image
from string_art.optimization import LoggingCallback, PlottingCallback


def run(img: np.ndarray, config: Config):
    img = preprocess_image(img, config.low_res, config.invert_input)
    mask = create_circular_mask(config.low_res)

    # Precompute string matrices
    A_high_res, A_low_res, valid_edges_mask = load_string_matrices(config.n_pins, config.pin_side_length, config.string_thickness,
                                                                   config.min_angle, config.high_res, config.low_res)

    # find/load optimal edges to approximate the image
    importance_map = np.ones((config.low_res, config.low_res))
    importance_map[~mask] = 0
    _, axs = plt.subplots(1, 2, figsize=(12, 7))
    axs[0].imshow(img)
    plt.show(block=False)
    callbacks = [LoggingCallback(n_edges=A_high_res.shape[1])]
    if config.plot_optimization:
        callbacks.append(PlottingCallback(axs[1], config.n_pins, config.high_res, config.pin_side_length, config.string_thickness))
    x = load_picked_edges(config.name_of_the_run, img, importance_map, A_high_res, A_low_res, valid_edges_mask, callbacks)
    plt.show()

    # reconstruct image
    recon = np.clip(A_high_res @ x, 0, 1)
    recon_image_high = recon.reshape(config.high_res, config.high_res)
    recon_image_high = np.flipud(recon_image_high.T)
    recon_image_low = matlab_imresize(recon_image_high, output_shape=(config.low_res, config.low_res))

    if config.invert_output:
        recon_image_high = 1 - recon_image_high
        recon_image_low = 1 - recon_image_low

    rmse_value = np.sqrt(np.mean((img[mask] - recon_image_low[mask])**2))
    print('RMSE: ', rmse_value)

    load_error_image(config.name_of_the_run, img, recon_image_low)
    # load_consecutive_path(name_of_the_run, x, config.n_pins, config.low_resolution,)
