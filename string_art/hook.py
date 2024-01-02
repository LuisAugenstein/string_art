import numpy as np
from scipy.spatial import ConvexHull
from string_art.line import Line
from typing import Literal


class Hook:
    corner_points: np.ndarray
    """np.shape(4, 2) array of the corner points of the hook in pixels"""

    def __init__(self, width: float, pos2d: np.ndarray, rotZAngleRadians: float) -> None:
        """
        Parameters
        -
        width: the width of the hook in pixels
        position: np.shape[2] array of the position of the hook in pixels
        angle: the Z-axis rotation angle of the hook in radians
        """
        self.width = width
        self.pos2d = pos2d
        self.rotZAngleRadians = rotZAngleRadians
        self.corner_points = self.__init_corner_points(width, pos2d, rotZAngleRadians)

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
        zero_width_diagonal: Line between two hook corners which does not account for any string width
        """
        alpha = np.arccos(string_width / zero_width_diagonal.length)
        if turn == 'left':
            alpha *= -1
        rotation = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        direction = rotation @ zero_width_diagonal.direction * string_width / 2
        return Line((zero_width_diagonal.start + direction, zero_width_diagonal.end - direction))

    def compute_strings(self, hookB: 'Hook', string_width=1.0) -> list[Line]:
        """
        computes the 4 possible string connections between two hooks A and B. 
        """
        a_points, b_points = self.corner_points, hookB.corner_points

        best_length_AB = None
        best_length_BA = None
        best_length_DiagAB = None
        best_length_DiagBA = None

        for i, b_point in enumerate(b_points):
            k = ConvexHull(np.vstack([a_points, b_point]))
            b_index = np.where(k.vertices == len(a_points))[0][0]
            jump_vertices = [k.vertices[(b_index + offset) % len(k.vertices)] for offset in [-1, 1]]
            a_to_B_index, B_to_a_index = jump_vertices
            lineAB = Line(np.vstack([k.points[a_to_B_index], b_point]))
            lineBA = Line(np.vstack([b_point, k.points[B_to_a_index]]))

            rem_B = np.delete(b_points, i, axis=0)

            line = lineAB

            distances_B = line.signed_distance(rem_B)
            dir_AB = line.end - line.start
            length_AB = np.linalg.norm(dir_AB)

            if np.all(distances_B >= 0) or np.all(distances_B <= 0):
                # AB is tangent to Hook B, test if it is AB or DiagAB
                rem_A = np.delete(a_points, a_to_B_index, axis=0)
                distances_A = line.distance(rem_A)

                if np.sum(distances_A >= 0) == np.sum(distances_B >= 0):
                    # AB
                    if best_length_AB is None or best_length_AB > length_AB:
                        best_length_AB = length_AB
                        AB = Line(line.end_points - 0.5 * string_width * line.normal[None, :])
                elif best_length_DiagAB is None or best_length_DiagAB > length_AB:
                    # DiagAB
                    DiagAB = self.get_string_width_diagonal(line, string_width, turn='left')
                    best_length_DiagAB = length_AB

            # Analyze BA
            line = lineBA

            distances_B = line.signed_distance(rem_B)
            dir_BA = line.end - line.start
            length_BA = np.linalg.norm(dir_BA)

            if np.all(distances_B >= 0) or np.all(distances_B <= 0):
                # BA is tangent to Hook B, test if it is BA or DiagBA
                rem_A = np.delete(a_points, B_to_a_index, axis=0)
                distances_A = line.distance(rem_A)
                if np.sum(distances_A >= 0) == np.sum(distances_B >= 0):
                    # BA
                    if best_length_BA is None or best_length_BA > length_BA:
                        best_length_BA = length_BA
                        BA = Line(line.end_points - 0.5 * string_width * line.normal[None, :])
                elif best_length_DiagBA is None or best_length_DiagBA > length_BA:
                    # DiaBA
                    DiagBA = self.get_string_width_diagonal(line, string_width, turn='right')
                    best_length_DiagBA = length_BA

        return [AB, BA, DiagAB, DiagBA]

    def intersects_string(self, p1: np.ndarray, p2: np.ndarray, threshold=1e-8) -> bool:
        """
        Parameters
        -
        p1: np.shape[2] array of the first point of the string segment
        p2: np.shape[2] array of the second point of the string segment

        Returns
        -
        True if the string segment intersects the hook, False otherwise
        """
        line = Line((p1, p2))
        distances = line.distance(self.corner_points)
        return np.any(distances <= threshold)

    def angle_to_x_axis(self, v: np.ndarray) -> float:
        return -np.arctan2(v[1], v[0])
