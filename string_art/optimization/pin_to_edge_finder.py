import numpy as np


class PinToEdgeFinder:
    def __init__(self, edge_codes: np.ndarray, connection_type: np.ndarray = None):
        """
        edge_codes: [4*n_edges, 2]
        """
        self.edge_codes = edge_codes
        self.connection_type = connection_type

    def compute(self, hook_index):
        i, j = np.where(self.edge_codes == hook_index)
        T = i % 4

        # if the desired hook occurs on the right side of the edge, switch connectiontype 0 and 1
        switch_mask = j == 1
        first_mask = switch_mask & (T == 0)
        second_mask = switch_mask & (T == 1)

        T[first_mask] = 1
        T[second_mask] = 0

        result = np.vstack([i, T]).T
        return result
