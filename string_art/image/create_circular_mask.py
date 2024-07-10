import torch


def create_circular_mask(image_size: int, radius: float = None) -> torch.Tensor:
    """
    Returns
    -
    mask: [image_size, image_size], dtype=torch.bool) mask with True values inside the circle and False values outside the circle
    """
    rows, cols = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')  # [image_size, image_size]
    grid = torch.stack([rows, cols], dim=-1).reshape(-1, 2)  # [image_size**2, 2]
    center = (image_size-1) // 2
    if radius is None:
        radius = center
    mask = (grid - center).pow(2).sum(-1) <= radius ** 2
    return mask.reshape(image_size, image_size)
