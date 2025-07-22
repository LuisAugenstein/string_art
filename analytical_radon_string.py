import torch
import numpy as np
import string_art.edges as edges
from string_art.image import create_circular_mask
import string_art.pins as pins
from string_art import line_profile
import matplotlib.pyplot as plt
from skimage.transform import radon
from radon_skimage import custom_radon
import time

IMAGE_SIZE = 512
N_RADON_ANGLES = 512 
STRING_WIDTH = 2
STRING_COLOR = 0.8
N_PINS = 16

pins_angle_based = pins.angle_based(N_PINS)  # [N_pins]
pins_point_based = pins.point_based(pins_angle_based, IMAGE_SIZE)  # [N_pins, 2]
edges_index_based = edges.index_based(N_PINS)  # [N_strings, 2]
edges_point_based = edges.point_based(pins_point_based, edges_index_based)  # [N_pins, 2, 2]
edges_angle_based = edges.angle_based(pins_angle_based, edges_index_based)  # [N_strings, 2]
radon_angles_radians = torch.arange(N_RADON_ANGLES) * torch.pi / N_RADON_ANGLES  # [N_RADON_ANGLES]
radon_angles_degrees = torch.arange(N_RADON_ANGLES) * 180 / N_RADON_ANGLES
circular_mask = create_circular_mask(IMAGE_SIZE, radius=IMAGE_SIZE//2 - 5)  # [IMAGE_SIZE, IMAGE_SIZE]
s_indices, alpha_indices = edges.radon_parameter_to_radon_index_based(edges_angle_based, radon_angles_radians, IMAGE_SIZE).T  # [N_strings] [N_strings]

edge_index = 22
edge_image = edges.get_image(edges_point_based[edge_index], line_profile.trapez(STRING_WIDTH, STRING_COLOR), IMAGE_SIZE)
edge_image[~circular_mask] = 0.

start_time = time.time()
sinogram = radon(edge_image, radon_angles_degrees)
print("run 1: ", time.time() - start_time)

analytic_radon = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
s_0, alpha_0_index = s_indices[edge_index], alpha_indices[edge_index]
alpha_0 = radon_angles_radians[alpha_0_index]


start_time = time.time()
# analytic_radon = custom_radon(edge_image.numpy(), radon_angles_radians.numpy())
# analytic_radon = (1 / torch.sin(torch.abs(radon_angles_radians - alpha_0))).unsqueeze(0).repeat(IMAGE_SIZE, 1) # [IMAGE_SIZE, N_RADON_ANGLES]
# analytic_radon[:, alpha_0_index] = 0.
# analytic_radon[s_0, alpha_0_index] = 1.
r_squared = (IMAGE_SIZE // 2)**2
for i, s in enumerate(range(IMAGE_SIZE)):
    for j, alpha in enumerate(radon_angles_radians):
        if s**2 + s_0**2 - 2*s*s_0 * np.cos(alpha - alpha_0) / np.sin(alpha - alpha_0)**2 > r_squared:
            analytic_radon[i,j] = 1 / np.sin(alpha - alpha_0)


print("run 2: ", time.time() - start_time)

fig, [ax_img, ax_skimage_radon, ax_analytic_radon] = plt.subplots(1, 3, figsize=(15, 5))

ax_img.set_title("Original image")
ax_img.set_ylabel("row r [pixels]")
ax_img.set_xlabel("column c [pixels]")
ax_img.imshow(edge_image, cmap='gray')

ax_skimage_radon.set_title("Radon transform\n(Sinogram)")
ax_skimage_radon.set_xlabel("Projection angle alpha [degrees]")
ax_skimage_radon.set_ylabel("distance from lower circle half [pixels]")
ax_skimage_radon.imshow(sinogram, cmap='hot')
ax_skimage_radon.set_xticks([0, 90, 180, 270, 360, 450, 512])
ax_skimage_radon.set_xticklabels([0, 30, 60, 90, 120, 150, 180])

ax_analytic_radon.set_title("analytic Radon transform\n(Sinogram)")
ax_analytic_radon.set_xlabel("Projection angle alpha [degrees]")
ax_analytic_radon.set_ylabel("distance from lower circle half [pixels]")
ax_analytic_radon.imshow(analytic_radon, cmap='hot')
ax_analytic_radon.set_xticks([0, 90, 180, 270, 360, 450, 512])
ax_analytic_radon.set_xticklabels([0, 30, 60, 90, 120, 150, 180])

plt.show()
