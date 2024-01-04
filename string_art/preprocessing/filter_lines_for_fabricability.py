import numpy as np
from string_art.preprocessing.get_pins import circular_pin_positions
from string_art.preprocessing.get_edges import get_edges
from string_art.entities import Pin, Line


def filter_lines_for_fabricability(possible_lines: list[Line], pins: list[Pin], min_angle: float, thresh=1e-8) -> tuple[list[Line], np.ndarray]:
    """
    Parameters
    -
    pins: [n_pins]      list of pin objects
    lines: [4*n_edges]  list of all possible lines between the pins
    min_angle: float    minimum allowed angle between two pins

    Returns
    -
    fabricable: np.shape([4*n_edges]) boolean mask indicating which strings are fabricable
    """
    n_pins = len(pins)
    inflated_edges = get_edges(n_pins).repeat(4, axis=0)
    fabricable = np.ones(inflated_edges.shape[0], dtype=bool)

    # do any neighbors of the pins intersect the line?
    for i, (edge, line) in enumerate(zip(inflated_edges, possible_lines)):
        neighbors = [neighbor for j in edge for neighbor in get_pin_neighbors(pins, j)]
        do_edge_neighbors_intersect_line = any([neighbor.intersects_string(line) for neighbor in neighbors])
        if do_edge_neighbors_intersect_line:
            fabricable[i] = False
            print('not fabricable ', line.start, line.end)
            raise ValueError('neighbor intersects.')

    # is angle between pins smaller than min_angle?
    edge_angles = get_edge_angles(n_pins, inflated_edges)
    fabricable[edge_angles - min_angle < thresh] = False
    return [line for i, line in enumerate(possible_lines) if fabricable[i]], fabricable


def get_edge_angles(n_pins: int, edges: np.ndarray) -> np.ndarray:
    pin_positions, _ = circular_pin_positions(n_pins, radius=1)
    p1 = pin_positions[edges[:, 0]]
    p2 = pin_positions[edges[:, 1]]
    edge_angles = np.squeeze(np.arccos(p1[:, None, :] @ p2[:, :, None]))
    return edge_angles


def get_pin_neighbors(pins: list[Pin], pin_index: int) -> tuple[Pin, Pin]:
    """
    Returns
    -
    [neighbor_a, neighbor_b]: list[Pin]  the two neighbor pins of the pin with the given index
    """
    neighbor_a_index = (pin_index + 1) % len(pins)
    neighbor_b_index = (pin_index - 1) % len(pins)
    return pins[neighbor_a_index], pins[neighbor_b_index]
