import numpy as np

Line = np.ndarray
"""np.shape([2, 2], dtype=int) first row equals start and second row the end point of the line"""

Lines = np.ndarray
"""np.shape([N, 2, 2]) list of lines"""


def direction_vector(line: Line) -> np.ndarray:
    start, end = line
    return (end - start) / length_of_line(line)


def length_of_line(line: Line) -> float:
    start, end = line
    return np.linalg.norm(end - start)


def normal_vector(line: Line) -> np.ndarray:
    dir = direction_vector(line)
    return np.array([-dir[1], dir[0]])


def signed_distance(line: Line, points: np.ndarray, thresh=1e-8) -> np.ndarray:
    start, _ = line
    distances = (points - start[None, :]) @ normal_vector(line)
    distances[np.abs(distances) < thresh] = 0
    return distances


def distance(line: Line, points: np.ndarray, thresh=1e-8) -> np.ndarray:
    """
    Parameters
    -
    points: np.shape[N, 2] array of points
    """
    return np.abs(signed_distance(line, points, thresh))
