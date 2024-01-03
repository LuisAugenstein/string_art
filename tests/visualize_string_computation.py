import matplotlib.pyplot as plt
from string_art.preprocessing import get_all_possible_pin_connections, get_pins, circular_pin_positions, lines_to_strings_in_positive_domain, filter_string_boundaries
from string_art.plotting import StringPresenter, plot_lines, plot_pins, plot_strings

n_pins = 8
pin_side_length = 2
string_thickness = 0.15
high_resolution = 1024

pins = get_pins(n_pins, radius=0.5*high_resolution, width=pin_side_length /
                string_thickness, pin_position_function=circular_pin_positions)
lines = get_all_possible_pin_connections(pins)
strings = lines_to_strings_in_positive_domain(lines, high_resolution)
strings = filter_string_boundaries(strings, high_resolution)

fig, axs = plt.subplots(1, 2, figsize=(15, 9))


def update_plot(line_idx=0):
    ax1, ax2 = axs
    ax1.clear()
    plot_pins(ax1, pins)
    plot_lines(ax1, lines[:line_idx])

    ax2.clear()
    plot_pins(ax2, pins, offset=0.5*high_resolution)
    plot_strings(ax2, strings[:line_idx], high_resolution)
    plt.draw()


presenter = StringPresenter(fig, update_plot, len(lines))
update_plot()
plt.show()
