import numpy as np
from matplotlib.axes import Axes
from string_art.entities import Pin, Line, String
from matplotlib.lines import Line2D


def plot_pins(ax: Axes, hook_array: list[Pin], offset=0., colored_hook_indices=[]) -> None:
    for i, hook in enumerate(hook_array):
        hookPos = np.vstack([hook.corner_points, hook.corner_points[0]]) + np.ones((5, 2)) * offset
        ax.plot(hookPos[:, 0], hookPos[:, 1], 'red' if i in colored_hook_indices else 'black')
        ax.set_aspect('equal', adjustable='box')


def plot_line(ax: Axes, line: Line) -> Line2D:
    return ax.plot(line[:, 0], line[:, 1], 'black')[0]


def plot_lines(ax: Axes, lines: list[Line]) -> list[Line2D]:
    return [plot_line(ax, line) for line in lines]


def plot_strings(ax: Axes, strings: list[String],  resolution: int) -> None:
    image = np.zeros((resolution, resolution))
    for string in strings:
        x, y, v = string
        image[x, y] = np.clip(image[x, y]+v, 0, 1)
    ax.imshow(1-image, cmap='gray', vmin=0, vmax=1)
