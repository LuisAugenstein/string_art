import numpy as np
from skimage.transform import warp

def custom_radon(image: np.ndarray, alpha: np.ndarray):
    """
    image: [IMAGE_SIZE, IMAGE_SIZE]  assume image is 0 outside the reconstruction circle
    alpha: [N_RADON_ANGLES] in radians
    """
    IMAGE_SIZE = image.shape[0]
    N_RADON_ANGLES = alpha.shape[0]
    radius = IMAGE_SIZE // 2

    radon_image = np.zeros((IMAGE_SIZE, N_RADON_ANGLES), dtype=image.dtype)

    for j, angle in enumerate(alpha):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array(
            [
                [cos_a, sin_a, -radius * (cos_a + sin_a - 1)],
                [-sin_a, cos_a, -radius * (cos_a - sin_a - 1)],
                [0, 0, 1],
            ]
        )
        rotated = warp(image, R, clip=False)
        radon_image[:, j] = rotated.sum(0)
    return radon_image


angle = np.pi/2
image = np.zeros((5,5))
image[1,2] = 1#
cos_a, sin_a = np.cos(angle), np.sin(angle)
radius = image.shape[0] // 2
R = np.array(
    [
        [cos_a, sin_a, -radius * (cos_a + sin_a - 1)],
        [-sin_a, cos_a, -radius * (cos_a - sin_a - 1)],
        [0, 0, 1],
    ]
)
rotated = warp(image, R, clip=False)

print(image)
print("")
print(np.round(rotated))
