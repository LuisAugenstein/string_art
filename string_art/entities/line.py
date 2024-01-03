import numpy as np
from skimage import draw

String = np.ndarray
"""np.shape([N, 3]) an anti-aliased line from along a pixel grid
The first two columns contain the integer x,y coordinates of the pixels
the third column contains the string intensity value of the pixel between 0 and 1"""


class Line:
    """
    high level string object representing the start and end points of a string
    """

    end_points: np.ndarray
    """np.shape[2, 2] first row equals start and second row the end point of the line"""

    def __init__(self, end_points: np.ndarray) -> None:
        self.end_points = end_points

    @property
    def start(self) -> np.ndarray:
        return self.end_points[0]

    @property
    def end(self) -> np.ndarray:
        return self.end_points[1]

    @property
    def direction(self) -> np.ndarray:
        return (self.end - self.start) / self.length

    @property
    def length(self) -> float:
        return np.linalg.norm(self.end - self.start)

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def signed_distance(self, points: np.ndarray, thresh=1e-8) -> np.ndarray:
        distances = (points - self.start[None, :]) @ self.normal
        distances[np.abs(distances) < thresh] = 0
        return distances

    def distance(self, points: np.ndarray, thresh=1e-8) -> np.ndarray:
        """
        Parameters
        -
        points: np.shape[N, 2] array of points
        """
        return np.abs(self.signed_distance(points, thresh))

    def reverse(self) -> 'Line':
        return Line(self.end_points[::-1])

    @staticmethod
    def empty() -> 'Line':
        return Line(np.zeros((2, 2)))

    def to_string(self) -> String:
        start, end = np.round(self.end_points).astype(np.int32)
        x, y = draw.line(start[0], start[1], end[0], end[1])
        v = np.ones_like(x)
        return np.vstack([x, y, v]).T
