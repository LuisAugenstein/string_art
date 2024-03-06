import torch
import numpy as np
from typing import Literal


class Pin:
    corner_points: torch.Tensor
    """torch.shape(4, 2) array of the corner points of the pin in pixels"""

    def __init__(self, pos2d: torch.Tensor, angle: float, width: float) -> None:
        """
        Parameters
        -
        pos2d: torch.shape[2] x,y pixel coordinates of the pin position
        angle: the Z-axis rotation angle of the pin in radians
        width: the width of the pin in pixels
        """
        self.pos2d = pos2d
        self.rotZAngleRadians = angle
        self.width = width
        self.corner_points = self.__init_corner_points(width, pos2d, angle)

    def __init_corner_points(self, width: float, pos2d: torch.Tensor, angle: float) -> torch.Tensor:
        base_corners = 0.5 * torch.Tensor([
            [-width, width],
            [-width, -width],
            [width, -width],
            [width, width]
        ])  # [4, 2]
        assert torch.allclose(torch.stack([pos2d + rotation_matrix(angle) @ corner for corner in base_corners]),
                              pos2d.unsqueeze(0) + base_corners @ rotation_matrix(angle).T)
        return pos2d.unsqueeze(0) + base_corners @ rotation_matrix(angle).T

    def get_string_width_diagonal(self, zero_width_diagonal: torch.Tensor, string_width: float, turn: Literal['left', 'right'] = 'left') -> torch.Tensor:
        """
        adds an offset to the given line to account for the string width

        Parameters
        - 
        zero_width_diagonal: torch.shape([2,2]) line between two pin corners which does not account for any string width

        Returns
        -
        diagonal: torch.shape([2,2]) line between two pin corners which accounts for the string width
        """
        start, end = zero_width_diagonal
        diagonal_length = torch.norm(end - start)
        angle = torch.arccos(string_width / diagonal_length)
        if turn == 'left':
            angle *= -1
        direction = rotation_matrix(angle) @ (end-start)/diagonal_length * string_width / 2
        return torch.vstack([start + direction, end - direction])

    def get_possible_connections(self, pinB: 'Pin', string_width=1.0) -> torch.Tensor:
        """
        computes the 4 possible string connections between two pins A and B by connecting the specific pin corners.
        """
        lines = torch.zeros((4, 2, 2))  # AB, BA, DiagAB, DiagBA
        best_lengths = [None, None, None, None]
        a_points, b_points = self.corner_points, pinB.corner_points

        for i, b_point in enumerate(b_points):
            distances = torch.norm((a_points - b_point[None, :]), axis=1)
            angles = self.__get_normalized_angles(b_point)
            thresh = 1e-8

            i_max_angles = torch.where(torch.abs(angles - torch.max(angles)) < thresh)[0]
            i_max_angle = i_max_angles[torch.argmin(distances[i_max_angles])]
            lineAB = torch.vstack([a_points[i_max_angle], b_point])

            i_min_angles = torch.where(torch.abs(angles - torch.min(angles)) < thresh)[0]
            i_min_angle = i_min_angles[torch.argmin(distances[i_min_angles])]
            lineBA = torch.vstack([b_point, a_points[i_min_angle]])

            jump_vertices = [i_max_angle, i_min_angle]
            rem_B = torch.cat(b_points[:i], b_points[i+1:])

            for j, line in enumerate([lineAB, lineBA]):
                distances_B = distance(line, rem_B, signed=True)
                start, end = line
                dir = end - start
                length = torch.norm(dir)
                if torch.all(distances_B >= 0) or torch.all(distances_B <= 0):
                    rem_A = torch.cat(a_points[:jump_vertices[j]], a_points[jump_vertices[j]+1:])
                    distances_A = distance(line, rem_A)
                    if torch.sum(distances_A >= 0) == torch.sum(distances_B >= 0):
                        if best_lengths[j] is None or best_lengths[j] > length:
                            best_lengths[j] = length
                            lines[j] = line - 0.5 * string_width * normal_vector(line)[None, :]
                    elif best_lengths[2+j] is None or best_lengths[2+j] > length:
                        turns = ['left', 'right']
                        lines[2+j] = self.get_string_width_diagonal(line, string_width, turns[j])
                        best_lengths[2+j] = length
        return lines

    def __get_normalized_angles(self, base_point: torch.Tensor) -> torch.Tensor:
        """
        Computes the relative angles from the given base_point to the corners of the pin.

        Parameters
        -
        base_point: torch.shape([2]) x,y pixel coordinates of the base point

        Returns
        -
        angles: [0, a1, a2, a3]   angles in radians. first angle is always 0. The remaining 3 angles are relative to the first angle.
        """
        diffs = base_point[:, None] - self.corner_points.T  # (2,4)
        first_angle = torch.arctan2(diffs[1, 0], diffs[0, 0])
        diff_rot = rotation_matrix(first_angle).T @ diffs
        return torch.arctan2(diff_rot[1, :], diff_rot[0, :])


def normal_vector(line: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    line: torch.shape([2, 2]) first row equals start and second row the end point of the line

    Returns
    -
    normal_vector: torch.shape([2]) 2D unit length normal vector of the line
    """
    start, end = line
    direction_vector = torch.nn.functional.normalize(end - start)
    return torch.Tensor([-direction_vector[1], direction_vector[0]])


def distance(line: torch.Tensor, points: torch.Tensor, signed=False, thresh=1e-8) -> torch.Tensor:
    """
    Parameters
    -
    line: torch.shape([2, 2]) first row equals start and second row the end point of the line
    points: torch.shape([N, 2]) array of points

    Returns
    -
    signed_distances: torch.shape([N]) signed distances of the points to the line. Negative distances are on the right side of the line
    """
    start, _ = line
    signed_distances = (points - start[None, :]) @ normal_vector(line)
    signed_distances[torch.abs(signed_distances) < thresh] = 0
    return signed_distances if signed else torch.abs(signed_distances)


def rotation_matrix(angle: float) -> torch.Tensor:
    """
    Returns
    -
    rotation_matrix: torch.shape([2, 2]) 2D rotation matrix
    """
    return torch.Tensor([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
