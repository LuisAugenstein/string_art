import torch
import numpy as np
from typing import Protocol
from dataclasses import dataclass
from skimage.transform import radon
from string_art.analytical_radon_line import analytical_radon_line
import string_art.pins as pins
import string_art.edges as edges


torch.set_default_dtype(torch.float64)


@dataclass
class StringReconstructionRadonConfig:
    n_pins: int = 300
    n_radon_angles: int = 150
    n_max_steps: int = 6000
    residual_threshold: float = 0.01
    line_darkness: float = 0.018
    p_min: float = 0.00008
    t_start: float = 0.0000005
    t_end: float = 0.000065


@dataclass
class StringReconstructionRadonCallbackConfig:
    step: int
    """current iteration starting at 0"""
    reconstructed_line_radon_index_based: tuple[int, int]
    """(s_index, alpha_index) indices of the radon parameters of the line chosen in the current iteration in their corresponding domains"""
    reconstructed_line_radon_parameter_based: tuple[float, float]
    """(s, alpha) radon parameters of the line chosen in the current iteration"""
    img_radon: np.ndarray
    """current residual radon image after subtracting the radon transformed reconstruced line"""
    residual: float
    """the maximum value of the radon transformed residual, i.e., the radon transformed image minus the radon transformed string reconstruction."""


class StringReconstructionRadonCallback(Protocol):
    def __call__(self, config: StringReconstructionRadonCallbackConfig) -> None:
        ...


def string_reconstruction_radon(img: np.ndarray, config: StringReconstructionRadonConfig, callback: StringReconstructionRadonCallback) -> None:
    """
    Parameters
    -
    img: np.ndarray  [Imagesize, Imagesize] values between 0="empty area" and 1="area where a string should be drawn"
    config: StringReconstructionRadonConfig configuration for the string reconstruction 
    callback: Custom callback function called in each iteration. Might be used for plotting or logging
    """
    IMAGE_SIZE = img.shape[0]
    # Image crop
    x = np.linspace(-1, 1, IMAGE_SIZE)
    y = np.linspace(-1, 1, IMAGE_SIZE)
    X, Y = np.meshgrid(x, y)
    img[(X ** 2 + Y ** 2) + 0.01 > 1] = 0

    # setup pins and edges in the different necessary representations
    pins_angle_based = pins.angle_based(config.n_pins)  # [N_pins]
    edges_index_based = edges.index_based(config.n_pins)  # [N_strings, 2]
    edges_angle_based = edges.angle_based(pins_angle_based, edges_index_based)  # [N_strings, 2]
    alpha_domain = torch.arange(config.n_radon_angles) * torch.pi / config.n_radon_angles  # [N_RADON_ANGLES]
    # alpha_domain_degrees = torch.arange(N_RADON_ANGLES) * 180 / N_RADON_ANGLES # TODO: check why linspace works but arange doesn't
    s_domain = torch.linspace(-1, 1, IMAGE_SIZE)
    edges_radon_parameter_based = edges.radon_parameter_based(edges_angle_based)  # [N_strings, 2]
    s_indices, alpha_indices = edges.radon_index_based(edges_radon_parameter_based, s_domain, alpha_domain).T  # [N_strings] [N_strings]
    valid_radon_parameters_mask = torch.zeros(IMAGE_SIZE, config.n_radon_angles, dtype=torch.bool)
    valid_radon_parameters_mask[s_indices, alpha_indices] = True
    # radon_beam_lengths = 2 * ((torch.sqrt(1 - (1 - torch.linspace(0, 2, IMAGE_SIZE)) ** 2) *
    #                            IMAGE_SIZE).unsqueeze(1).repeat(1, config.n_radon_angles)).numpy() / IMAGE_SIZE  # [IMAGE_SIZE, N_RADON_ANGLES]
    # radon_beam_lengths[radon_beam_lengths == 0] = torch.inf

    s_domain = s_domain.numpy()

    alpha_deg = np.linspace(0, 180, config.n_radon_angles)
    alpha_domain = np.linspace(0, np.pi, config.n_radon_angles)
    img_radon = radon(img, alpha_deg)
    img_radon = img_radon / IMAGE_SIZE

    ALPHA, S = np.meshgrid(alpha_domain, s_domain)  # [N_RADON_ANGLES, IMAGE_SIZE]
    line_lengths = 2 * np.sqrt(1 - S ** 2)
    zero_length_mask = line_lengths == 0
    # we are intereseted in the radon value per unit length
    img_radon[~zero_length_mask] = img_radon[~zero_length_mask] / line_lengths[~zero_length_mask]
    img_radon[zero_length_mask] = 0
    # remove all lines that cannot be spanned between two pins
    img_radon[~valid_radon_parameters_mask] = 0

    for step in range(config.n_max_steps):
        s_index, alpha_index = np.unravel_index(np.argmax(img_radon), img_radon.shape)
        residual = img_radon[s_index, alpha_index]
        if residual < config.residual_threshold:
            print('Optimization finished')
            return

        s, alpha = s_domain[s_index], alpha_domain[alpha_index]
        p_line_theory = analytical_radon_line(alpha, s, ALPHA, S, line_lengths, config.t_start, config.t_end, config.line_darkness, config.p_min)
        img_radon = img_radon - p_line_theory
        img_radon[s_index, alpha_index] = 0

        callback(StringReconstructionRadonCallbackConfig(step, (s_index, alpha_index), (s, alpha), img_radon, residual))
