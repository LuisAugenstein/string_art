import numpy as np
from string_art import line_profile
import string_art.edges as edges
from string_art.image import create_circular_mask
import string_art.pins as pins
import torch
from matlab.analytic_radon_line import analytic_radon_line
import matplotlib.pyplot as plt
from skimage.transform import radon

N_PINS = 300
IMAGE_SIZE = 400
STRING_WIDTH = 0.5
STRING_COLOR = 1

pins_angle_based = pins.angle_based(N_PINS)  # [N_pins]
pins_point_based = pins.point_based(pins_angle_based, IMAGE_SIZE)  # [N_pins, 2]
edges_index_based = edges.index_based(N_PINS)  # [N_strings, 2]
edges_point_based = edges.point_based(pins_point_based, edges_index_based)  # [N_pins, 2, 2]
edges_angle_based = edges.angle_based(pins_angle_based, edges_index_based)  # [N_strings, 2]
radon_angles_radians = torch.arange(N_PINS) * torch.pi / N_PINS  # [N_RADON_ANGLES]
radon_angles_degrees = torch.arange(N_PINS) * 180 / N_PINS
circular_mask = create_circular_mask(IMAGE_SIZE, radius=IMAGE_SIZE//2 - 5)  # [IMAGE_SIZE, IMAGE_SIZE]

edge_index = 100
edge_image = edges.get_image(edges_point_based[edge_index], line_profile.trapez(STRING_WIDTH, STRING_COLOR), IMAGE_SIZE)
edge_image[~circular_mask] = 0.

img_radon = radon(edge_image, theta=radon_angles_degrees)

tstart = 0.0000005
tend = 0.000065
d = 0.018
p_min = 0.00008
R = 1

s = np.linspace(-R, R, N_PINS)
alpha = np.linspace(0, 180, N_PINS)
ALPHA, S = np.meshgrid(alpha, s)
L = 2 * np.sqrt(R ** 2 - S ** 2)
analytical_radon_transform = analytic_radon_line(alpha[edge_index], s[edge_index], ALPHA, S, R, L, tstart, tend, d, p_min)


fig, [ax_img, ax_radon, ax_radon_analytic] = plt.subplots(1, 3)
ax_img.imshow(edge_image, cmap='gray')
radon_plot = ax_radon.imshow(img_radon, cmap='hot')
plt.colorbar(radon_plot)
analytic_radon_plot = ax_radon_analytic.imshow(analytical_radon_transform, cmap='hot')
plt.colorbar(analytic_radon_plot)


plt.show()
