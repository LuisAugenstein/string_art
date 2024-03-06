import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
from string_art.entities import place_pins, circular_pin_positions
from string_art.plotting import plot_pins, plot_line
from matplotlib.pyplot import Axes
from scipy.sparse import csc_matrix
from string_art.transformations import indices_1D_to_2D


class PlottingCallback():
    def __init__(self, ax: Axes, A_high_res: csc_matrix, n_pins: int, pin_side_length: float, string_thickness: float) -> None:
        self.mode = 'add'
        if ax is None:
            self.fig = plt.figure(figsize=(12, 7))
            ax = self.fig.gca()
        else:
            self.fig = ax.get_figure()
        high_res, n_strings = int(np.sqrt(A_high_res.shape[0])), A_high_res.shape[1]
        pins = place_pins(n_pins, radius=0.5*high_res, width=pin_side_length /
                          string_thickness, pin_position_function=circular_pin_positions)
        plot_pins(ax, pins, offset=0.5*high_res)
        self.lines = np.array([[indices_1D_to_2D(A_high_res[:, j].indices[0], high_res),
                                indices_1D_to_2D(A_high_res[:, j].indices[-1], high_res)] for j in range(n_strings)]).squeeze()
        self.line_plots = {}

    def choose_next_string(self, step: int, i_next_string: int | None, __f_score: float | None) -> None:
        if i_next_string is None:
            return
        ax = self.fig.gca()
        if self.mode == 'add':
            line_plot = plot_line(ax, self.lines[i_next_string])
            self.line_plots[i_next_string] = line_plot
        elif self.mode == 'remove':
            line_plot = self.line_plots.pop(i_next_string)
            line_plot.remove()
        plt.pause(0.001)

    def switch_mode(self, new_mode: Literal['add', 'remove']) -> None:
        self.mode = new_mode
