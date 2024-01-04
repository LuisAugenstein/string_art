from string_art.entities import Pin, Line
from string_art.preprocessing.get_edges import get_edges


def get_all_possible_pin_connections(pins: list[Pin]) -> list[Line]:
    """
    Parameters
    -
    pins:         list of pin objects

    Returns
    -
    lines: [4*n_edges]  list of line objects representing the strings between all possible pairs of pins.
                        n_edges = n_choose_k(n_pins, 2)
    """
    n_pins = len(pins)
    return [line for i, j in get_edges(n_pins) for line in pins[int(i)].get_possible_connections(pins[j])]
