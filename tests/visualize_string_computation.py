import matplotlib.pyplot as plt
from string_art.entities import place_pins
from string_art.preprocessing import get_edges, edges_to_lines_in_positive_domain
from string_art.transformations import line_to_string
from string_art.plotting import StringPresenter, plot_lines, plot_pins, plot_strings

n_pins = 8
pin_side_length = 2
string_thickness = 0.15
high_res = 1024

pins = place_pins(n_pins, radius=0.5*high_res, width=pin_side_length / string_thickness)
edges = get_edges(n_pins)
lines = edges_to_lines_in_positive_domain(pins, edges, high_res)
strings = [line_to_string(line) for line in lines]


fig, axs = plt.subplots(1, 2, figsize=(15, 9))


def update_plot(line_idx=0):
    ax1, ax2 = axs
    ax1.clear()
    plot_pins(ax1, pins, offset=0.5*high_res)
    plot_lines(ax1, lines[:line_idx])

    ax2.clear()
    plot_pins(ax2, pins, offset=0.5*high_res)
    plot_strings(ax2, strings[:line_idx], high_res)
    plt.draw()


presenter = StringPresenter(fig, update_plot, len(lines))
update_plot()
plt.show()
