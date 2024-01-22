import numpy as np
from string_art.entities.line import Line, Lines, direction_vector, length_of_line, signed_distance, distance, normal_vector
from typing import Literal


class Pin:
    corner_points: np.ndarray
    """np.shape(4, 2) array of the corner points of the pin in pixels"""

    def __init__(self, position: np.ndarray, angle: float, width: float) -> None:
        """
        Parameters
        -
        width: the width of the pin in pixels
        position: np.shape[2] array of the position of the pin in pixels
        angle: the Z-axis rotation angle of the pin in radians
        """
        self.pos2d = position
        self.rotZAngleRadians = angle
        self.width = width
        self.corner_points = self.__init_corner_points(width, position, angle)

    def __init_corner_points(self, width: float, pos2d: np.ndarray, angle: float) -> np.ndarray:
        base_corners = 0.5 * np.array([
            [-width, width],
            [-width, -width],
            [width, -width],
            [width, width]
        ])
        rotMatrix = np.array([[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])
        return np.array([pos2d + rotMatrix @ corner for corner in base_corners])

    def get_string_width_diagonal(self, zero_width_diagonal: Line, string_width: float, turn: Literal['left', 'right'] = 'left') -> Line:
        """
        adds an offset to the given line to account for the string width

        Parameters
        - 
        zero_width_diagonal: Line between two pin corners which does not account for any string width
        """
        alpha = np.arccos(string_width / length_of_line(zero_width_diagonal))
        if turn == 'left':
            alpha *= -1
        rotation = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        direction = rotation @ direction_vector(zero_width_diagonal) * string_width / 2
        start, end = zero_width_diagonal
        return np.vstack([start + direction, end - direction])

    def get_possible_connections(self, pinB: 'Pin', string_width=1.0) -> np.ndarray:
        """
        computes the 4 possible string connections between two pins A and B by connecting the specific pin corners.
        """
        lines = np.zeros((4, 2, 2))  # AB, BA, DiagAB, DiagBA
        best_lengths = [None, None, None, None]
        a_points, b_points = self.corner_points, pinB.corner_points

        for i, b_point in enumerate(b_points):
            distances = np.linalg.norm((a_points - b_point[None, :]), axis=1)
            angles = self.__get_normalized_angles(b_point)
            thresh = 1e-8

            i_max_angles = np.where(np.abs(angles - np.max(angles)) < thresh)[0]
            i_max_angle = i_max_angles[np.argmin(distances[i_max_angles])]
            lineAB = np.vstack([a_points[i_max_angle], b_point])

            i_min_angles = np.where(np.abs(angles - np.min(angles)) < thresh)[0]
            i_min_angle = i_min_angles[np.argmin(distances[i_min_angles])]
            lineBA = np.vstack([b_point, a_points[i_min_angle]])

            jump_vertices = [i_max_angle, i_min_angle]
            rem_B = np.delete(b_points, i, axis=0)

            for j, line in enumerate([lineAB, lineBA]):
                distances_B = signed_distance(line, rem_B)
                start, end = line
                dir = end - start
                length = np.linalg.norm(dir)
                if np.all(distances_B >= 0) or np.all(distances_B <= 0):
                    rem_A = np.delete(a_points, jump_vertices[j], axis=0)
                    distances_A = distance(line, rem_A)
                    if np.sum(distances_A >= 0) == np.sum(distances_B >= 0):
                        if best_lengths[j] is None or best_lengths[j] > length:
                            best_lengths[j] = length
                            lines[j] = line - 0.5 * string_width * normal_vector(line)[None, :]
                    elif best_lengths[2+j] is None or best_lengths[2+j] > length:
                        turns = ['left', 'right']
                        lines[2+j] = self.get_string_width_diagonal(line, string_width, turns[j])
                        best_lengths[2+j] = length
        return lines

    def __get_normalized_angles(self, base_point: np.ndarray) -> np.ndarray:
        """
        Computes the relative angles from the given base_point to the corners of the pin.

        Returns
        -
        angles: [0, a1, a2, a3]   angles in radians. first angle is always 0. The remaining 3 angles are relative to the first angle.
        """
        diffs = base_point[:, None] - self.corner_points.T  # (2,4)
        first_angle = np.arctan2(diffs[1, 0], diffs[0, 0])
        rot_mat_inverse = np.array([[np.cos(first_angle), -np.sin(first_angle)],
                                    [np.sin(first_angle), np.cos(first_angle)]]).T
        diff_rot = rot_mat_inverse @ diffs
        return np.arctan2(diff_rot[1, :], diff_rot[0, :])
