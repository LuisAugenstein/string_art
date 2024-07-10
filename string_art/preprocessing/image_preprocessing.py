import torch
from string_art.transformations import matlab_imresize


def preprocess_image(img: torch.Tensor, resolution: int, invert: bool) -> torch.Tensor:
    """
    Parameters
    -
    img: torch.shape([N, N])  square input image with grayscale values between 0 and 255
    resolution: int             resolution of the output image
    invert: bool                whether to invert the image or not. Choos such that the background is light and the content dark.

    Returns
    -
    img: torch.shape([resolution, resolution]) resized, normalized and (inverted) image with values between 0 and 1
    """
    img = img / 255
    img = matlab_imresize(img, output_shape=(resolution, resolution))
    img = torch.clip(img, 0, 1)
    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    if invert:
        img = 1 - img
    return torch.flipud(img)
