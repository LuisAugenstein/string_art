import numpy as np
from string_art.entities import String, Pin, Lines, get_pins, circular_pin_positions
from string_art.transformations import strings_to_sparse_matrix, draw_line
from itertools import combinations
from tqdm import tqdm
from scipy.sparse import csr_matrix


def precompute_string_matrix(n_pins: int, pin_side_length: float, string_thickness: float, min_angle: float, high_res: int) -> tuple[csr_matrix, np.ndarray]:
    print('\n===Precompute String Matrix===')
    pins = get_pins(n_pins, radius=0.5*high_res, width=pin_side_length/string_thickness)
    edges = get_edges(n_pins)  # [n_edges, 2]
    print(f'n_pins={n_pins}, n_edges={edges.shape[0]}')
    fabricable = get_fabricability_mask(edges, n_pins, min_angle)  # [4*n_edges]
    print(f'min_angle={min_angle:.4f} exludes {np.sum(~fabricable)} edges.')
    lines = edges_to_lines_in_positive_domain(pins, edges, high_res)  # [4*n_edges]
    lines -= 1  # account for 0 indexing opposed for 1 indexing in matlab
    print(f'Compute A_high_res for high_res={high_res}')
    high_res_strings = [filter_string_boundaries(draw_line(line), high_res) for line in tqdm(lines)]
    A_high_res = strings_to_sparse_matrix(high_res_strings, high_res)
    print(f'A_high_res.shape={A_high_res.shape[0]}x{A_high_res.shape[1]}')
    return A_high_res, fabricable


def get_edges(n_pins: int) -> np.ndarray:
    """
    Returns
    edges: np.shape([n_edges, 2]) (i,j) where i,j are the indices of the pins
    """
    return np.array(list(combinations(range(n_pins), 2)))


def get_fabricability_mask(edges: np.ndarray, n_pins: int, min_angle: float, thresh=1e-8) -> np.ndarray:
    """
    Parameters
    -
    edges: np.shape([n_edges, 2])  list of all possible lines between the pins
    n_pins: int         number of pins
    min_angle: float    minimum allowed angle between two pins

    Returns
    -
    fabricable: np.shape([n_edges]) boolean mask indicating which edges are fabricable
    """
    fabricable = np.ones(edges.shape[0], dtype=bool)
    pin_positions, _ = circular_pin_positions(n_pins, radius=1)  # [n_pins, 2]
    p1, p2 = pin_positions[edges[:, 0]], pin_positions[edges[:, 1]]  # [n_edges, 2], [n_edges, 2]
    dot_p = np.sum(p1*p2, axis=1)  # [n_edges]
    edge_angles = np.squeeze(np.arccos(np.clip(dot_p, -1, 1)))
    fabricable[edge_angles - min_angle < thresh] = False
    return fabricable.repeat(4)


def edges_to_lines_in_positive_domain(pins: list[Pin], edges: np.ndarray, high_res: int) -> Lines:
    return np.array([np.round(line + (high_res+1)/2).astype(np.int32) for i, j in edges for line in pins[int(i)].get_possible_connections(pins[j])])


def filter_string_boundaries(string: String, high_res) -> String:
    x, y, v = string
    filter_string_boundary_mask = (x >= 0) & (x < high_res) & (y >= 0) & (y < high_res) & (v > 0)
    return x[filter_string_boundary_mask], y[filter_string_boundary_mask], v[filter_string_boundary_mask]
