import numpy as np
from math import comb
from itertools import combinations
from scipy.sparse import find
from string_art.entities import ConnectionType


class PinEdgeTransformer:
    def __init__(self, n_pins: int, valid_edges_mask: np.ndarray | None = None):
        """
        n_pins: int       number of pins
        fabricable_edges: np.shape([n_edges], dtype=bool) with n_edges=4*comb(4,2) binary mask indicating which edges are valid and which should be excluded.
                          if None then all edges are valid.
        """
        self.n_pins = n_pins
        n_edges = len(ConnectionType)*comb(n_pins, 2)
        self.valid_edges_mask = np.ones(n_edges, dtype=bool) if valid_edges_mask is None else valid_edges_mask
        self.__all_edges_to_pins = self.__init_all_edges_to_pins(n_pins, self.valid_edges_mask)
        self.__all_pins_to_edges = self.__init_all_pins_to_edges(n_pins,  self.__all_edges_to_pins)

    def pins_to_edges(self, pin_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        -
        pin_indices: np.shape([N], dtype=int) indices of N pins. N is at most n_pins.

        Returns
        -
        edge_indices: np.shape([N, n_incident_edges], dtype=int) each pin has n_incident_edges edges that connect to it. 
        connection_types: np.shape([N, n_incident_edges], dtype=ConnectionType) the connection type of each edge
                      n_incident_edges is at most 4*(n_pins-1) if all edges are valid
        """
        edge_indices, connection_types = self.__all_pins_to_edges[pin_indices, :, 0], self.__all_pins_to_edges[pin_indices, :, 1]
        return edge_indices, connection_types

    def edges_to_pins(self, edge_indices: np.ndarray) -> np.ndarray:
        """
        Parameters
        -
        edge_indices: np.shape([N], dtype=int) indices of N edges
                      valid values are between 0 and sum(valid_edges_mask) which is at most 4*comb(n_pins, 2).

        Returns
        -
        pin_indices: np.shape([N, 2], dtype=int) the same N edges but represented by their start and end pin indices
                     valid valus are between 0 and n_pins
        """
        return self.__all_edges_to_pins[edge_indices]

    def __init_all_edges_to_pins(self, n_pins: int, valid_edges_mask: np.ndarray) -> np.ndarray:
        all_edges_to_pins = np.array(list(combinations(range(n_pins), 2)))
        all_edges_to_pins = np.repeat(all_edges_to_pins, len(ConnectionType), axis=0)[valid_edges_mask]
        return all_edges_to_pins

    def __init_all_pins_to_edges(self, n_pins: int, all_edges_to_pins: np.ndarray) -> np.ndarray:
        all_pins_to_edges = []
        for pin_index in range(n_pins):
            _, i, _ = find(all_edges_to_pins[:, 0] == pin_index)
            _, i2, _ = find(all_edges_to_pins[:, 1] == pin_index)
            # zeros where pin_index is the first pin, ones where it is the second pin
            is_ingoing_edge = np.concatenate([np.zeros_like(i), np.ones_like(i2)])
            i = np.concatenate([i, i2])  # indices of edges that contain pin_index

            # if the desired hook occurs on the right side of the edge, switch connectiontype 0 and 1
            connection_type = i % 4
            connection_type[is_ingoing_edge & (connection_type == ConnectionType.STRAIGHT_IN)] = ConnectionType.STRAIGHT_OUT
            connection_type[is_ingoing_edge & (connection_type == ConnectionType.STRAIGHT_OUT)] = ConnectionType.STRAIGHT_IN

            all_pins_to_edges.append(np.vstack([i, connection_type]).T)
        return np.vstack(all_pins_to_edges).T
