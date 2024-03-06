import numpy as np
import matplotlib.pyplot as plt
from string_art.io import load_picked_edges, load_string_matrices, load_error_image
from string_art.transformations import matlab_imresize
from string_art.config import Config
from string_art.preprocessing import create_circular_mask, preprocess_image
from string_art.optimization import LoggingCallback, PlottingCallback, IterativeGreedyOptimizer, StringSelection, OptimizedLoss, SimpleLoss
from string_art.api import get_np_array_module_bool
import torch
from time import time


def run(img: np.ndarray, config: Config):
    torch.set_default_dtype(torch.float64)
    img: torch.Tensor = torch.Tensor(img)

    if config.use_cuda:
        if not torch.cuda.is_available():
            print("WARN: cuda is not available. Falling back to cpu computation.")
        torch.set_default_device('cuda')
        img = img.cuda()

    img = preprocess_image(img, config.low_res, config.invert_input)
    mask = create_circular_mask(config.low_res)
    importance_map = torch.ones((config.low_res, config.low_res))
    importance_map[~mask] = 0

    # Precompute string matrices
    A_high_res, A_low_res, valid_edges_mask = load_string_matrices(config.n_pins, config.pin_side_length, config.string_thickness,
                                                                   config.min_angle, config.high_res, config.low_res)

    img = img.cpu().numpy()
    mask = mask.cpu().numpy()
    importance_map = importance_map.cpu().numpy()

    # Run optimization or load edges from disk
    axs = __plot_image(img)
    callbacks = [LoggingCallback(n_edges=A_high_res.shape[1])]
    if config.plot_optimization:
        callbacks.append(PlottingCallback(axs[1], A_high_res, config.n_pins, config.pin_side_length, config.string_thickness))
    xp, xipy = get_np_array_module_bool(config.use_cuda)
    losses = {
        'simple-loss': lambda: SimpleLoss(xp.array(img), xp.ones_like(importance_map), xipy.sparse.csc_matrix(A_high_res), np.sqrt(A_low_res.shape[0]).astype(int)),
        'optimized-loss': lambda: OptimizedLoss(xp.array(img), xp.ones_like(importance_map), xipy.sparse.csc_matrix(A_high_res), xipy.sparse.csc_matrix(A_low_res))
    }
    optimizer = IterativeGreedyOptimizer(losses[config.loss_type](), StringSelection(valid_edges_mask), callbacks)
    x = load_picked_edges(config.name_of_the_run, optimizer)

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


def __plot_image(img: np.ndarray) -> list[plt.Axes]:
    _, axs = plt.subplots(1, 2, figsize=(12, 7))
    axs[0].imshow(img)
    plt.show(block=False)
    return axs
