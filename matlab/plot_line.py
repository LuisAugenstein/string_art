import numpy as np
from matplotlib.axes import Axes


def plot_line(ax: Axes, alpha, s, transparency):
    psi_1, psi_2 = alpha - np.arccos(s), alpha + np.arccos(s)

    start_x, start_y = np.cos(psi_1), np.sin(psi_1)
    end_x, end_y = np.cos(psi_2), np.sin(psi_2)

    ax.plot([start_x, end_x], [start_y, end_y], color=[0, 0, 0, transparency])
