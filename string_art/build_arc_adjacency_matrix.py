from string_art.config import Config
from string_art.drawLine import filter_string_boundaries
import numpy as np
from string_art.pin import Pin
from itertools import combinations, accumulate
from string_art.line import Line, String
import matplotlib.pyplot as plt


def get_possible_connections(pins: list[Pin]) -> list[Line]:
    """
    Parameters
    -
    pins:         list of pin objects

    Returns
    -
    lines:        list of line objects representing the strings between all possible pairs of pins
    fabricatable: boolean mask indicating which strings are fabricatable 
    Both list have length 4*n_edges with n_edges = n_choose_k(n_pins, 2)
    """
    n_pins = len(pins)
    edges = combinations(range(n_pins), 2)  # [n_edges,2]
    lines: list[Line] = []
    for edge in edges:
        pin_a, pin_b = [pins[i] for i in edge]
        possible_connections = pin_a.get_possible_connections(pin_b)
        neighbors = [neighbor for pin_index in edge for neighbor in get_pin_neighbors(pins, pin_index)]
        for line in possible_connections:
            is_fabricable = not any([neighbor.intersects_string(line) for neighbor in neighbors])
            if not is_fabricable:
                print('not fabricable ', line.start, line.end)
            lines.append(line if is_fabricable else None)
    return lines


def get_pin_neighbors(pins: list[Pin], pin_index: int) -> tuple[Pin, Pin]:
    """
    Returns
    -
    [neighbor_a, neighbor_b]: list[Pin]  the two neighbor pins of the pin with the given index
    """
    neighbor_a_index = (pin_index + 1) % len(pins)
    neighbor_b_index = (pin_index - 1) % len(pins)
    return pins[neighbor_a_index], pins[neighbor_b_index]


def circular_pin_positions(n_pins: int, radius: float) -> np.ndarray:
    """
    places n points around a circle with radius r

    Returns
    -
    pin_positions: np.shape([n_pins, 2])  x,y coordinates of the pins
    angles: np.shape([n_pins])  angles of the pins spanning a circle. can be used for spherical coordinates (angle, radius)
    """
    pin_angles = np.linspace(0, 2*np.pi, n_pins, endpoint=False)
    pin_positions = np.column_stack([np.cos(pin_angles), np.sin(pin_angles)])
    return radius*pin_positions, pin_angles


def get_pins(n_pins: int, radius, width: float, pin_position_function=circular_pin_positions) -> list[Pin]:
    pin_positions, pin_angles = pin_position_function(n_pins, radius)
    return [Pin(pos, angle, width) for pos, angle in zip(pin_positions, pin_angles)]


def lines_to_strings_in_positive_domain(lines: list[Line], domain_width: float) -> list[String]:
    strings: list[String] = []
    for i, line in enumerate(lines):
        if line is None:
            strings.append(None)
        line = Line(np.round(line.end_points + 0.5 * domain_width))
        # string = filter_string_boundaries(line.to_string(), domain_min=0, domain_max=domain_width-1)  # [n_string_pixels, 3]
        string = line.to_string()
        strings.append(string)
    return strings


def string_to_string_1D(string: String, domain_width: float) -> np.ndarray:
    x, y, v = string.T
    pixel_code_in_positive_domain = y * domain_width + x
    return np.vstack([pixel_code_in_positive_domain, v]).T


def list_to_array(arrays: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Parameters
    -
    arrays: [np.shape([N_i, ...])]    list of numpy arrays of different lengths

    Returns
    -
    concatenated_array: np.shape([N, ...])   array containing all the elements of the input arrays
    trajectory_indices: [(start, end)]       indices indicating the start and end of each array in the concatenated_array
    """
    lengths = list(accumulate([x.shape[0] for x in arrays]))
    trajectory_indices = zip(lengths[:-1], lengths[1:])
    return np.vstack(arrays), trajectory_indices
