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


def point_based(N_pins: int) -> torch.Tensor:
    """
    Parameters
    -
    pin_angles: [N_pins]  angles of the pins in radians

    Returns
    -
    pins_point_based: [N_pins, 2])  x, y coordinates of the pins starting at (1, 0) and going counter-clockwise
    """
    pins_angle_based = angle_based(N_pins)
    return torch.column_stack([torch.cos(pins_angle_based), torch.sin(pins_angle_based)])
