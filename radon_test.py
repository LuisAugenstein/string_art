import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import radon, rescale

from string_art.image import create_circular_mask, load_input_image
import torch

IMAGE_SIZE = 512
STRING_WIDTH = 0.5
STRING_COLOR = 0.8
N_PINS = 16
TARGET_IMAGE_PATH = 'data/inputs/smiley.png'

image = load_input_image(TARGET_IMAGE_PATH, IMAGE_SIZE).mean(dim=-1)
image[:, :IMAGE_SIZE//2] = 1.
circular_mask = create_circular_mask(IMAGE_SIZE, radius=IMAGE_SIZE//2 - 5)  # [IMAGE_SIZE, IMAGE_SIZE]
# circular_mask2 = create_circular_mask(IMAGE_SIZE, radius=IMAGE_SIZE//2 + 5)  # [IMAGE_SIZE, IMAGE_SIZE]
# image[~circular_mask2] = 0.
image = 1 - image
image[~circular_mask] = 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original image")
ax1.set_ylabel("row r [pixels]")
ax1.set_xlabel("column c [pixels]")
ax1.imshow(image, cmap=plt.cm.Greys_r, extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0))

theta = torch.arange(max(image.shape)) * 180 / max(image.shape)
sinogram = torch.tensor(radon(image, theta))
print(torch.max(sinogram))
print(torch.min(sinogram))
# ax2.set_title("Radon transform\n(Sinogram)")
# ax2.set_xlabel("Projection angle alpha [degrees]")
# ax2.set_ylabel("distance from lower circle half [pixels]")
# ax2.imshow(
#     sinogram,
#     cmap='gray',
#     extent=(0, 180, IMAGE_SIZE, 0),
#     aspect='auto',
# )

# fig.tight_layout()
# plt.show()
