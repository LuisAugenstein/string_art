import torch

type PinAngleBased = torch.Tensor
"""[1] angle representation in radians. 0='Pin at the right', pi/4='Pin at the top'"""
type PinsAngleBased = torch.Tensor
"""[N, 1]"""
type PinPointBased = torch.Tensor
"""[2] point representation in x-y coordinates. (1, 0)='pin at the right', (0, 1)='pin at the top'"""
type PinsPointBased = torch.Tensor
"""[N, 2]"""

def angle_based(N_pins: int) -> PinsAngleBased:
    """
    **Parameters**  
    N_pins: number of pins
    
    **Returns**  
    pins_angle_based: [N_pins]
    """
    return torch.arange(N_pins) * 2 * torch.pi / N_pins # [N_pins]


def point_based(N_pins: int) -> PinsPointBased:
    """
    **Parameters**  
    N_pins: number of pins
    
    **Returns**  
    pins_point_based: [N_pins, 2]
    """
    pins_angle_based = angle_based(N_pins)
    return torch.column_stack([torch.cos(pins_angle_based), torch.sin(pins_angle_based)])
