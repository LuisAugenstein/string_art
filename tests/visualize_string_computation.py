import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
from string_art import Line, Pin, plot_pins, plot_lines, plot_strings, get_possible_connections, get_pins, circular_pin_positions
from string_art.build_arc_adjacency_matrix import lines_to_strings_in_positive_domain, list_to_array

n_pins = 8
pin_side_length = 2
string_thickness = 0.15
high_resolution = 1024

pins = get_pins(n_pins, radius=0.5*high_resolution, width=pin_side_length /
                string_thickness, pin_position_function=circular_pin_positions)
lines = get_possible_connections(pins)
strings = lines_to_strings_in_positive_domain(lines, high_resolution)


fig, axs = plt.subplots(1, 2, figsize=(15, 9))
line_idx = 0
timer = False


def update_plot():
    ax1, ax2 = axs
    ax1.clear()
    plot_pins(ax1, pins)
    plot_lines(ax1, lines[:line_idx])

    ax2.clear()
    plot_pins(ax2, pins, offset=0.5*high_resolution)
    plot_strings(ax2, strings[:line_idx])
    plt.draw()


def on_key(event: KeyEvent):
    global timer, line_idx, lines
    if event.key == 'right':
        line_idx = (line_idx + 1) % len(lines)
    elif event.key == 'left':
        line_idx = (line_idx - 1) % len(lines)

    if timer:
        return

    timer = fig.canvas.new_timer(interval=1)
    timer.add_callback(update_plot)
    timer.start()
    print(f'Key pressed: {event.key}')


def on_key_release(event: KeyEvent):
    global timer
    if not timer:
        return
    timer.stop()
    timer = None
    print(f'Key released: {event.key}')


fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('key_release_event', on_key_release)
update_plot()
plt.show()
