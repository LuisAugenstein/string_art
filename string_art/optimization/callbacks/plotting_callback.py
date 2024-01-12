from typing import Literal
import matplotlib.pyplot as plt
from string_art.preprocessing import get_pins, circular_pin_positions, get_all_possible_pin_connections
from string_art.plotting import plot_pins, plot_line


class PlottingCallback():
    def __init__(self, n_pins: int, high_res: int, pin_side_length: float, string_thickness: float, plot_interval=1) -> None:
        self.plot_interval = plot_interval
        self.mode = 'add'
        self.fig = plt.figure(figsize=(12, 7))
        ax = self.fig.gca()
        pins = get_pins(n_pins, radius=0.5*high_res, width=pin_side_length /
                        string_thickness, pin_position_function=circular_pin_positions)
        plot_pins(ax, pins)
        self.lines = get_all_possible_pin_connections(pins)
        self.line_plots = {}

    def choose_next_edge(self, step: int, i_next_edge: int | None, __f_score: float | None) -> None:
        if step % self.plot_interval != 0 or i_next_edge is None:
            return
        ax = self.fig.gca()
        if self.mode == 'add':
            line_plot = plot_line(ax, self.lines[i_next_edge])
            self.line_plots[i_next_edge] = line_plot
        elif self.mode == 'remove':
            line_plot = self.line_plots.pop(i_next_edge)
            line_plot.remove()
        plt.pause(0.001)

    def switch_mode(self, new_mode: Literal['add', 'remove']) -> None:
        self.mode = new_mode
