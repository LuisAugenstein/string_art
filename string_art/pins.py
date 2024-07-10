import torch


def angle_based(N_pins: int) -> torch.Tensor:
    """
    Parameters
    -
    N_pins: int  number of pins

    Returns
    -
    pin_angles: [N_pins]  angles of the pins in radians
    """
    return torch.arange(N_pins) * 2 * torch.pi / N_pins


def point_based(pins_angle_based: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    Parameters
    -
    pin_angles: [N_pins]  angles of the pins in radians

    Returns
    -
    pins_point_based: [N_pins, 2])  row, col coordinates of the pins starting at (0, 1) and going counter-clockwise
    """
    centered_unit_circle = torch.column_stack([-torch.sin(pins_angle_based), torch.cos(pins_angle_based)])
    first_quadrant_unit_circle = (centered_unit_circle + 1)
    radius = (image_size - 1) // 2
    pins_point_based = radius * first_quadrant_unit_circle
    return pins_point_based
