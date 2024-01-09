import numpy as np
from math import comb
from itertools import combinations
from scipy.sparse import find
from string_art.entities import ConnectionType, N_CONNECTION_TYPES, connection_type_masks
from typing import Literal


class PinEdgeTransformer:
    def __init__(self, n_pins: int, valid_edges_mask: np.ndarray | None = None):
        """
        n_pins: int       number of pins
        fabricable_edges: np.shape([n_edges], dtype=bool) with n_edges=4*comb(4,2) binary mask indicating which edges are valid and which should be excluded.
                          if None then all edges are valid.
        """
        self.n_pins = n_pins
        n_possible_edges = N_CONNECTION_TYPES*comb(n_pins, 2)
        self.valid_edges_mask = np.ones(n_possible_edges, dtype=bool) if valid_edges_mask is None else valid_edges_mask
        self.__all_edges_to_pins = self.__init_all_edges_to_pins(n_pins, self.valid_edges_mask)
        self.__all_pins_to_edges = self.__init_all_pins_to_edges(n_pins,  self.__all_edges_to_pins)

    def pins_to_edges(self, pin_indices: np.ndarray | None = None, filter: Literal['ingoing', 'outgoing'] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        -
        pin_indices: np.shape([N], dtype=int) indices of N pins. if pin_indices is None all pins are used.
        filter: 'ingoing'  only returns ingoing edges, i.e.,  edges of the form (_, pin_index)
                'outgoing' only returns outgoing edges, i.e., edges of the form (pin_index, _)

        Returns
        -
        edge_indices: np.shape([N, n_incident_edges], dtype=int) each pin has n_incident_edges edges that connect to it. 
        connection_types: np.shape([N, n_incident_edges], dtype=ConnectionType) the connection type of each edge
                      n_incident_edges is at most 4*(n_pins-1) if all edges are valid
        """
        if pin_indices is None:
            pin_indices = np.arange(self.n_pins)
        n_input_pins = pin_indices.shape[0]
        edge_indices, connection_types = self.__all_pins_to_edges[pin_indices, :, 0], self.__all_pins_to_edges[pin_indices, :, 1]
        mask = connection_type_masks[filter](connection_types) if filter else np.ones_like(connection_types, dtype=bool)
        return edge_indices[mask].reshape(n_input_pins, -1), connection_types[mask].reshape(n_input_pins, -1)

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
        all_edges_to_pins = np.repeat(all_edges_to_pins, N_CONNECTION_TYPES, axis=0)[valid_edges_mask]
        return all_edges_to_pins

    def __init_all_pins_to_edges(self, n_pins: int, all_edges_to_pins: np.ndarray) -> np.ndarray:
        all_pins_to_edges = []
        for pin_index in range(n_pins):
            _, i, _ = find(all_edges_to_pins[:, 0] == pin_index)
            _, i2, _ = find(all_edges_to_pins[:, 1] == pin_index)
            pin_to_edges = np.concatenate([i, i2])  # indices of edges that contain pin_index

            # if pin_index occurs on the right side of the edge, i.e., in an ingoing edge,
            # switch connectiontype STRAIGHT_IN: 0 and STRAIGHT_OUT: 1
            connection_types = pin_to_edges % 4
            is_ingoing_edge = np.concatenate([np.zeros_like(i, dtype=bool), np.ones_like(i2, dtype=bool)])
            ingoing_straight_mask = is_ingoing_edge & ((connection_types == ConnectionType.STRAIGHT_IN) |
                                                       (connection_types == ConnectionType.STRAIGHT_OUT))
            connection_types[ingoing_straight_mask] = 1 - connection_types[ingoing_straight_mask]
            all_pins_to_edges.append(np.vstack([pin_to_edges, connection_types]).T)
        return np.stack(all_pins_to_edges, axis=0)
