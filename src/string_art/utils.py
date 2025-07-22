import torch

def create_circular_mask(image_size: int, epsilon: float = 0.01) -> torch.Tensor:
    """
    Returns
    circular_mask: [image_size, image_size] boolean mask which is True outside the circle and False inside
    """
    x = torch.linspace(-1, 1, image_size)
    y = torch.linspace(-1, 1, image_size)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    return (X ** 2 + Y ** 2) + epsilon > 1
