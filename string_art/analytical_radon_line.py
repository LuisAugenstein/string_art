import numpy as np


def analytical_radon_line(alpha0, s0, ALPHA, S, L, tstart, tend, d, p_min):
    ALPHA_diff_alpha0 = ALPHA - alpha0
    sin_ALPHA_alpha0 = np.sin(ALPHA_diff_alpha0)
    sin_ALPHA_alpha0_squared = sin_ALPHA_alpha0 ** 2

    n = 4
    t = np.linspace(tstart, tend, n)
    nominator = (S ** 2 + s0 ** 2 - 2 * S * s0 * np.cos(ALPHA_diff_alpha0))
    p_regionfuns = [nominator / (sin_ALPHA_alpha0_squared + t_i) - 1 for t_i in t]

    mask = np.zeros_like(ALPHA)
    mask[p_regionfuns[0] < 0] = 1
    for i in range(n - 1):
        mask[(p_regionfuns[i] > 0) & (p_regionfuns[i+1] < 0)] = (tend - t[i]) / (tend - tstart)

    p_line = d * p_min / ((d * L - p_min) * np.abs(sin_ALPHA_alpha0) + p_min)
    return p_line * mask
