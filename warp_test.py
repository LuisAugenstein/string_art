import numpy as np
from skimage.transform import warp

IMAGE_SIZE = 6

image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
image[3, 2] = 1
print(image)

angle = np.pi
radius = IMAGE_SIZE // 2

cos_a, sin_a = np.cos(angle), np.sin(angle)
R = np.array(
    [
        [cos_a, sin_a, -radius * (cos_a + sin_a - 1)],
        [-sin_a, cos_a, -radius * (cos_a - sin_a - 1)],
        [0, 0, 1],
    ]
)
warped = warp(image, R)
print(np.round(warped))
