from string_art import Pin, Line, String
import numpy as np
from matplotlib.axes import Axes


def plot_pins(ax: Axes, hook_array: list[Pin], offset=0., colored_hook_indices=[]) -> None:
    for i, hook in enumerate(hook_array):
        hookPos = np.vstack([hook.corner_points, hook.corner_points[0]]) + np.ones((5, 2)) * offset
        ax.plot(hookPos[:, 0], hookPos[:, 1], 'red' if i in colored_hook_indices else 'black')
        ax.set_aspect('equal', adjustable='box')


def plot_lines(ax: Axes, lines: Line | list[Line]) -> None:
    if isinstance(lines, Line):
        lines = [lines]
    for l in lines:
        ax.plot([l.start[0], l.end[0]], [l.start[1], l.end[1]], 'blue')


def plot_strings(ax: Axes, strings: list[String]) -> None:
    pixels = {}
    for string in strings:
        for x, y, v in string:
            x, y = int(x), int(y)
            if (x, y) not in pixels:
                pixels[(x, y)] = 0
            pixels[(x, y)] = min(1, pixels[(x, y)]+v)

    points = np.zeros((len(pixels), 3))
    for i, ((x, y), v) in enumerate(pixels.items()):
        points[i, :] = np.array([x, y, v])

    ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='gray', s=0.1)
