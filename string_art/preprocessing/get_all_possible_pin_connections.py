from itertools import combinations
from string_art.entities import Pin, Line


def get_all_possible_pin_connections(pins: list[Pin]) -> list[Line]:
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
            # TODO: check if neighbor.intersects_string is even necessary. I never encountered this issue
            is_fabricable = not any([neighbor.intersects_string(line) for neighbor in neighbors])
            if not is_fabricable:
                print('not fabricable ', line.start, line.end)
                raise ValueError('not fabricable')
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
