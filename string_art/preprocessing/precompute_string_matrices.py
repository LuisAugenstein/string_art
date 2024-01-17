import numpy as np
from scipy.sparse import find
from string_art.entities import String
from string_art.preprocessing.filter_string_boundaries import filter_string_boundaries
from string_art.preprocessing.get_all_possible_pin_connections import get_all_possible_pin_connections
from string_art.preprocessing.lines_to_strings_in_positive_domain import lines_to_strings_in_positive_domain
from string_art.preprocessing.get_pins import get_pins
from string_art.transformations import strings_to_sparse_matrix, imresize
from string_art.preprocessing.get_fabricability_mask import get_fabricability_mask


def precompute_string_matrices(n_pins: int, pin_side_length: float, string_thickness: float, min_angle: float, high_res: int, low_res: float):
    pins = get_pins(n_pins, radius=0.5*high_res, width=pin_side_length/string_thickness)
    fabricable = get_fabricability_mask(pins, min_angle)
    connection_lines = get_all_possible_pin_connections(pins)
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
        # fast image resize
        low_res_image = image.reshape(low_res, high_res // low_res, low_res, high_res // low_res).mean(axis=(1, 3))

        # image resize like in the matlab code
        # low_res_image = imresize(image, output_shape=(low_res, low_res))
        low_res_string = find(low_res_image)
        low_res_strings.append(np.vstack(low_res_string).T)
    return low_res_strings
