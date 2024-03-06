import matplotlib.pyplot as plt
from string_art.entities import place_pins
from string_art.plotting import plot_pins, plot_strings, plot_strings, StringPresenter
from string_art.io import load_string_matrices
from string_art.transformations import sparse_matrix_to_strings
from math import pi

n_pins = 16
pin_side_length = 2
string_thickness = 0.15
high_resolution = 1024
min_angle = pi / 4
super_sampling_window_width = 4
low_resolution = high_resolution // super_sampling_window_width

pins = place_pins(n_pins, radius=0.5*high_resolution, width=pin_side_length / string_thickness)
A, B, _ = load_string_matrices(n_pins, pin_side_length, string_thickness, min_angle, high_resolution, low_resolution)
high_res_strings = sparse_matrix_to_strings(A)
low_res_strings = sparse_matrix_to_strings(B)

fig, axs = plt.subplots(1, 2, figsize=(15, 9))
low_res_pins = place_pins(n_pins, radius=0.5*low_resolution, width=pin_side_length / string_thickness / super_sampling_window_width)


def update_plot(line_idx=0):
    ax1, ax2 = axs
    ax1.clear()
    plot_pins(ax1, pins, offset=0.5*high_resolution)
    plot_strings(ax1, high_res_strings[:line_idx], high_resolution)

    ax2.clear()
    plot_pins(ax2, low_res_pins, offset=0.5*low_resolution)
    plot_strings(ax2, low_res_strings[:line_idx], low_resolution)
    plt.draw()


presenter = StringPresenter(fig, update_plot, len(high_res_strings))
update_plot()
plt.show()
