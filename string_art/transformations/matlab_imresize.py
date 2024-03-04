import torch
import numpy as np
"""
implementation taken from https://github.com/fatheral/matlab_imresize
"""


def cubic(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)
    absx = torch.absolute(x)
    absx2 = torch.multiply(absx, absx)
    absx3 = torch.multiply(absx2, absx)
    f = torch.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + torch.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f


def contributions(in_length: int, out_length: int, kernel=cubic, k_width=4.0) -> tuple[torch.Tensor, torch.Tensor]:
    scale = out_length / in_length
    x = torch.arange(1, out_length+1, dtype=torch.float64)
    u = (x + (scale - 1) / 2) / scale
    scale = np.clip(scale, -np.inf, 1.0)
    kernel_width = k_width / scale
    left = torch.floor(u - kernel_width / 2)

    indices = (left[:, None] + torch.arange(-1, np.ceil(kernel_width) + 1)).to(torch.int32)  # -1 because indexing from 0
    kernel_input = u.unsqueeze(1) - indices - 1  # -1 because indexing from 0
    weights = scale * kernel(scale * kernel_input)

    weights /= torch.sum(weights, dim=1)[:, None]
    aux = torch.concatenate((torch.arange(in_length), torch.arange(in_length - 1, -1, step=-1))).to(torch.int32)
    i_mod = torch.fmod(indices, aux.numel())
    indices = aux[i_mod]
    ind2store = torch.nonzero(torch.any(weights, axis=0))  # indices of the columns that contain at least one non-zero element
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizevec(in_img: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Parameters
    -
    in_img: torch.size([width, height])
    weights: torch.size([low_res, n_channels])
    indices: torch.size([low_res, n_channels])
    dim: 0 or 1

    Returns
    -
    out_img: torch.size([low_res, height]) for dim=0 
             torch.size([width, low_res])  for dim=1
    """
    if dim == 0:
        weights = weights.unsqueeze(-1)          # [low_res, n_channels, 1]
        img = in_img[indices]                    # [low_res, n_channels, height]
        out_img = torch.sum(weights*img, axis=1)  # [low_res, height]
    elif dim == 1:
        weights = weights.unsqueeze(0)           # [1, low_res, n_channels]
        img = in_img[:, indices]                 # [low_res, low_res, n_channels]
        out_img = torch.sum(weights*img, axis=2)  # [low_res, low_res]

    if in_img.dtype == torch.uint8:
        out_img = torch.clip(out_img, 0, 255)
        return torch.round(out_img).to(torch.uint8)
    else:
        return out_img


def matlab_imresize(img: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
    """
    Parameters
    -
    img: torch.shape([width, height])  input image with grayscale values between 0 and 1
    output_shape: torch.Size([new_width, new_height])  output shape of the image

    Returns
    -
    img: torch.shape([new_width, new_height]) resized image with values between 0 and 1
    """
    for k in range(2):
        weights, indices = contributions(img.shape[k], output_shape[k])
        img = imresizevec(img, weights.squeeze(), indices.squeeze(), k)
    return img
