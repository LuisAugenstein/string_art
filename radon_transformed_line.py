import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import radon, rescale
import string_art.line_profile as line_profile
import string_art.edges as edges
import string_art.pins as pins
from string_art.image import create_circular_mask, load_input_image
import torch

IMAGE_SIZE = 256
STRING_WIDTH = 0.5
STRING_COLOR = 1
N_PINS = 16
TARGET_IMAGE_PATH = 'data/inputs/smiley.png'

# image = load_input_image(TARGET_IMAGE_PATH, IMAGE_SIZE)
# image = torch.mean(image, dim=-1).numpy()
# image[:, :IMAGE_SIZE//2] = 1.
# circular_mask = create_circular_mask(IMAGE_SIZE, radius=IMAGE_SIZE//2 - 5)  # [IMAGE_SIZE, IMAGE_SIZE]
# image[~circular_mask] = 1.
# image = 1 - image

pin_positions = pins.point_based(N_PINS, IMAGE_SIZE)
edges_index_based = edges.index_based(N_PINS)
edges_point_based = edges.point_based(pin_positions, edges_index_based)
edge = edges_point_based[16]
image = edges.get_image(edge, line_profile.trapez(STRING_WIDTH, STRING_COLOR), IMAGE_SIZE)

circular_mask = create_circular_mask(IMAGE_SIZE, radius=IMAGE_SIZE//2 - 5)  # [IMAGE_SIZE, IMAGE_SIZE]
image[~circular_mask] = 0.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0.0, 180.0, max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)
sinogram = np.flip(sinogram, axis=0)
dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
ax2.set_title("flipped Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(
    sinogram,
    cmap=plt.cm.Greys_r,
    extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
    aspect='auto',
)

fig.tight_layout()
plt.show()
