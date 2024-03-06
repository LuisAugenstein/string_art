import numpy as np
import torch
from string_art.entities.pin import Pin


def circular_pin_positions(n_pins: int, radius: float) -> torch.Tensor:
    """
    places n points around a circle with radius r

    Returns
    -
    pin_positions: torch.shape([n_pins, 2])  x,y coordinates of the pins
    angles: torch.shape([n_pins])  angles of the pins spanning a circle. can be used for spherical coordinates (angle, radius)
    """
    pin_angles = torch.linspace(0, 2*np.pi - 2*np.pi/n_pins, n_pins)
    pin_positions = torch.column_stack([torch.cos(pin_angles), torch.sin(pin_angles)])
    return radius*pin_positions, pin_angles


def place_pins(n_pins: int, radius, width: float, pin_position_function=circular_pin_positions) -> list[Pin]:
    pin_positions, pin_angles = pin_position_function(n_pins, radius)
    return [Pin(pos, angle, width) for pos, angle in zip(pin_positions, pin_angles)]
