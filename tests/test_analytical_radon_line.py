import torch
import numpy as np
from string_art import line_profile
import string_art.edges as edges
import string_art.pins as pins
import matplotlib.pyplot as plt
from skimage.transform import radon
from string_art.string_reconstruction_radon import StringReconstructionRadonConfig
from string_art.utils import create_circular_mask
from string_art.analytical_radon_line import analytical_radon_line

IMAGE_SIZE = 400
STRING_WIDTH = 0.01
TRAPEZ_SLOPE = 200
config = StringReconstructionRadonConfig(line_darkness=0.7)

pins_angle_based = pins.angle_based(config.n_pins)  # [N_pins]
pins_point_based = pins.point_based(config.n_pins)  # [N_pins, 2]
edges_index_based = edges.index_based(config.n_pins)  # [N_strings, 2]
edges_point_based = edges.point_based(pins_point_based, edges_index_based)  # [N_pins, 2, 2]
radon_angles_radians = torch.arange(config.n_radon_angles) * torch.pi / config.n_radon_angles  # [N_RADON_ANGLES]
radon_angles_degrees = torch.arange(config.n_radon_angles) * 180 / config.n_radon_angles

edge_index = 149
edge = edges_point_based[edge_index]
edge_image = 1-edges.get_image(edge, line_profile.trapez(STRING_WIDTH, config.line_darkness, TRAPEZ_SLOPE), IMAGE_SIZE)
edge_image[create_circular_mask(IMAGE_SIZE)] = 0

alpha_domain_deg = np.linspace(0, 180, config.n_radon_angles)
img_radon = radon(edge_image, theta=alpha_domain_deg)

s_domain = np.linspace(-1, 1, config.n_radon_angles)
alpha_domain = np.linspace(0, np.pi, config.n_radon_angles)
_, S = np.meshgrid(alpha_domain, s_domain)
line_lengths = 2 * np.sqrt(1 - S ** 2)
alpha, s = alpha_domain[edge_index], s_domain[edge_index]
analytical_radon_transform = analytical_radon_line(alpha, s, alpha_domain, s_domain, line_lengths,
                                                   config.t_start, config.t_end, config.line_darkness, config.p_min)


fig, [ax_img, ax_radon, ax_radon_analytic] = plt.subplots(1, 3, figsize=(15, 7))
ax_img.set_title(f'Edge {edge_index}')
ax_img.imshow(edge_image, cmap='gray', extent=[0, IMAGE_SIZE, IMAGE_SIZE, 0])
radon_plot = ax_radon.imshow(img_radon, cmap='gray')
# plt.colorbar(radon_plot)
analytic_radon_plot = ax_radon_analytic.imshow(analytical_radon_transform, cmap='gray')
# plt.colorbar(analytic_radon_plot)

plt.show()
