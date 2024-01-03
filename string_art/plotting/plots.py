import numpy as np
from matplotlib.axes import Axes
from string_art.entities import Pin, Line, String


def plot_pins(ax: Axes, hook_array: list[Pin], offset=0., colored_hook_indices=[]) -> None:
    for i, hook in enumerate(hook_array):
        hookPos = np.vstack([hook.corner_points, hook.corner_points[0]]) + np.ones((5, 2)) * offset
        ax.plot(hookPos[:, 0], hookPos[:, 1], 'red' if i in colored_hook_indices else 'black')
        ax.set_aspect('equal', adjustable='box')


def plot_lines(ax: Axes, lines: Line | list[Line]) -> None:
    if isinstance(lines, Line):
        lines = [lines]
    for l in lines:
        ax.plot([l.start[0], l.end[0]], [l.start[1], l.end[1]], 'black')


def plot_strings(ax: Axes, strings: list[String],  resolution: int, s=1) -> None:
    image = np.zeros((resolution, resolution))
    for string in strings:
        for x, y, v in string:
            x, y = int(x), int(y)
            image[x, y] = min(1, image[x, y]+v)
    ax.imshow(1-image, cmap='gray', vmin=0, vmax=1)
