import numpy as np
from skimage.transform import resize
from scipy.sparse import csr_matrix, find
from string_art.entities import Pin, String
from string_art.preprocessing.filter_string_boundaries import filter_string_boundaries
from string_art.preprocessing.get_all_possible_pin_connections import get_all_possible_pin_connections
from string_art.preprocessing.lines_to_strings_in_positive_domain import lines_to_strings_in_positive_domain
from string_art.preprocessing.get_pins import get_pins
from string_art.transformations import strings_to_sparse_matrix
from string_art.preprocessing.filter_lines_for_fabricability import filter_lines_for_fabricability


def precompute_string_matrices(n_pins: int, pin_side_length: float, string_thickness: float, min_angle: float, high_res: int, low_res: float):
    pins = get_pins(n_pins, radius=0.5*high_res, width=pin_side_length/string_thickness)
    connection_lines = get_all_possible_pin_connections(pins)
    fabricable_lines, fabricable = filter_lines_for_fabricability(connection_lines, pins, min_angle)
    # strings = lines_to_strings_in_positive_domain(fabricable_lines, high_res)
    strings = lines_to_strings_in_positive_domain(connection_lines, high_res)
    strings = filter_string_boundaries(strings, high_res)

    high_res_matrix = strings_to_sparse_matrix(strings, high_res)
    low_res_strings = strings_to_lower_resolution(strings, high_res, low_res)
    low_res_matrix = strings_to_sparse_matrix(low_res_strings, low_res)
    return high_res_matrix, low_res_matrix, fabricable


def strings_to_lower_resolution(strings: list[String], high_res: int, low_res: int) -> list[String]:
    low_res_strings = []
    for step, string in enumerate(strings):
        print(f'{step+1}/{len(strings)} high to low resolution')
        image = np.zeros((high_res, high_res))
        x, y, v = string.T
        image[x, y] = v
        low_res_image = resize(image, (low_res, low_res), mode='constant')
        low_res_string = find(low_res_image)
        low_res_strings.append(np.vstack(low_res_string).T)
    return low_res_strings
