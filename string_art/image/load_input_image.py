import torch
import numpy as np
from PIL import Image


def load_input_image(path: str, image_size: int) -> torch.Tensor:
    """
    Parameters
    -
    path: file path to the image which should be loaded
    image_size: width/height of the square input image

    Returns
    -
    img: [image_size, image_size]   
    """
    img = Image.open(path)
    resized_img = img.resize((image_size, image_size))
    normalized_img = torch.tensor(np.array(resized_img)) / 255.  # [H, W] value 0=black to 1=white
    return normalized_img
