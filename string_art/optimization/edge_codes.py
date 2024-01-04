from itertools import combinations
from string_art.optimization.pin_to_edge_finder import PinToEdgeFinder
import numpy as np


def edge_codes(n_pins: int, fabricable: np.ndarray):
    edge_codes = np.array(list(combinations(range(n_pins), 2)))
    n_connections_between_pins = 4
    edge_codes = np.repeat(edge_codes, n_connections_between_pins, axis=0)[fabricable]
    n_edges = len(edge_codes)

    connection_type = np.tile(np.array(list(range(n_connections_between_pins))), n_edges)  # [0, 1, 2, 3, 0, 1, 2, 3]
    hook_to_edge_finder = PinToEdgeFinder(edge_codes, np.array(connection_type))
    hook_to_edge = [hook_to_edge_finder.compute(i) for i in range(n_pins)]
    # hook_to_edge tells us in which edge the hook i is present.
    # if the edge is (i,_) then the connection type (second column) is 0 1 2 3
    # if the edge is (_,i) then the connection type (second column) is 1 0 3 2
    # Note that this can probably be done in a simpler way later on. We don't need hook_to_edge at all
    # instead we just check (i,_) or (_,i) when using the edge

    return edge_codes, hook_to_edge
