from string_art import Hook, plot_hooks, plot_lines
import numpy as np
import matplotlib.pyplot as plt


def visualize_string_computation():
    hook_a = Hook(width=2, pos2d=np.array([0., 0.]), rotZAngleRadians=0)
    hook_b = Hook(width=2, pos2d=np.array([4., 3.]), rotZAngleRadians=0.9)

    lines = hook_a.compute_strings(hook_b, string_width=0.2)
    plot_hooks([hook_a, hook_b])
    plot_lines(lines)
    plt.show()


visualize_string_computation()
