import numpy as np
from string_art.entities import String, Pin, Lines, place_pins, circular_pin_positions
from string_art.transformations import strings_to_sparse_matrix
from string_art.preprocessing.xiaolinwu import xiaolinwu
from itertools import combinations
from scipy.sparse import csc_matrix
from string_art.utils import map
from tqdm import tqdm
import torch


def precompute_string_matrix(n_pins: int, pin_side_length: float, string_thickness: float, min_angle: float, high_res: int) -> tuple[csc_matrix, np.ndarray]:
    print('\n===Precompute String Matrix===')
    pins = place_pins(n_pins, radius=0.5*high_res, width=pin_side_length/string_thickness)
    edges = get_edges(n_pins)  # [n_edges, 2]
    print(f'n_pins={n_pins}, n_edges={edges.shape[0]}, n_strings={4*edges.shape[0]}')
    fabricable = get_fabricability_mask(edges, n_pins, min_angle)  # [n_strings]
    print(f'min_angle={min_angle:.4f} exludes {torch.sum(~fabricable)} strings.')
    print(f'Compute lines in positive domain')
    lines = edges_to_lines_in_positive_domain(pins, edges, high_res)  # [n_strings, 2, 2]

    lines = lines.cpu().numpy()

    lines -= 1  # account for 0 indexing opposed for 1 indexing in matlab
    print(f'Compute A_high_res for high_res={high_res}')
    high_res_strings = map(lambda line: filter_string_boundaries(xiaolinwu(line), high_res), lines, performance_mode=True)
    A_high_res = strings_to_sparse_matrix(high_res_strings, high_res)
    print(f'A_high_res.shape={A_high_res.shape[0]}x{A_high_res.shape[1]}')
    return A_high_res, fabricable


def get_edges(n_pins: int) -> torch.Tensor:
    """
    [ 0 0 ... 0        1 1 ... n_pins-2 ] \\
    [ 1 2 ... n_pins-1 2 3 ... n_pins-1 ] \\

    Returns
    edges: torch.shape([n_edges, 2]) (i,j) where i,j are the indices of the pins 
    """
    return torch.combinations(torch.arange(n_pins), r=2)


def get_fabricability_mask(edges: torch.Tensor, n_pins: int, min_angle: float, thresh=1e-8) -> torch.Tensor:
    """
    Parameters
    -
    edges: torch.shape([n_edges, 2])  list of all possible lines between the pins
    n_pins: int         number of pins
    min_angle: float    minimum allowed angle between two pins

    Returns
    -
    fabricable: torch.shape([4*n_edges]) boolean mask indicating which strings are fabricable
    """
    fabricable: torch.Tensor = torch.ones(edges.shape[0], dtype=bool)
    pin_positions, _ = circular_pin_positions(n_pins, radius=1)  # [n_pins, 2]
    p1, p2 = pin_positions[edges[:, 0]], pin_positions[edges[:, 1]]  # [n_edges, 2], [n_edges, 2]
    dot_p = torch.sum(p1*p2, axis=1)  # [n_edges]
    edge_angles = torch.arccos(torch.clip(dot_p, -1, 1)).squeeze()
    fabricable[edge_angles - min_angle < thresh] = False
    return fabricable.repeat_interleave(4)


def edges_to_lines_in_positive_domain(pins: list[Pin], edges: torch.Tensor, high_res: int) -> torch.Tensor:
    """
    Parameters
    -
    pins: list[Pin]         list of Pin objects
    edges: torch.shape([n_edges, 2])  list of all possible edges between the pins
    high_res: int           number of pixels in the positive domain

    Returns
    -
    lines: torch.shape([4*n_edges, 2, 2])  list of all possible lines between the pins
    """
    nested_lines = map(lambda edge: pins[edge[0]].get_possible_connections(pins[edge[1]]), edges)
    return torch.stack([torch.round(line + (high_res+1)/2).to(torch.int32) for lines in nested_lines for line in lines])


def filter_string_boundaries(string: String, high_res) -> String:
    x, y, v = string
    filter_string_boundary_mask = (x >= 0) & (x < high_res) & (y >= 0) & (y < high_res) & (v > 0)
    return x[filter_string_boundary_mask], y[filter_string_boundary_mask], v[filter_string_boundary_mask]
