from string_art import Hook, Line
import numpy as np
import matplotlib.pyplot as plt


def plot_hooks(hook_array: list[Hook], colored_hook_indices=[]) -> None:
    for i, hook in enumerate(hook_array):
        hookPos = np.vstack([hook.corner_points, hook.corner_points[0]])
        plt.plot(hookPos[:, 0], hookPos[:, 1], 'red' if i in colored_hook_indices else 'black')
        plt.gca().set_aspect('equal', adjustable='box')


def plot_lines(string_data: tuple[Line]) -> None:
    """
    Parameter
    -
    string_data: tuple[np.shape[2,2]] 
    """
    colors = ['cyan', 'magenta', 'green', 'blue']
    for p, color in zip(string_data, colors):
        p1, p2 = p.end_points
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color)
