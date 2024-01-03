import numpy as np


def XiaolinWu(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    -
    start_point: np.shape([2])
    end_point: np.shape([2])

    Returns
    -
    pixels: np.shape([N, 2])   an anti-aliased line from start_point to end_point along a pixel grid
    intensities: np.shape([N]) values between 0 and 1 representing the intensity of the pixels
    """
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1
    # preallocate memory for x, y, and c
    length = int(np.floor(3 * np.sqrt(dx**2 + dy**2)))
    x = np.zeros(length)
    y = np.zeros_like(x)
    c = np.zeros_like(x)

    swapped = False
    if np.abs(dx) < np.abs(dy):
        x1, y1 = swap(x1, y1)
        x2, y2 = swap(x2, y2)
        dx, dy = swap(dx, dy)
        swapped = True

    if x2 < x1:
        x1, x2 = swap(x1, x2)
        y1, y2 = swap(y1, y2)

    gradient = dy / dx

    # handle first endpoint
    xend = np.round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = rfpart(x1 + 0.5)
    xpxl1 = xend  # this will be used in the main loop
    ypxl1 = ipart(yend)
    x[0] = xpxl1
    y[0] = ypxl1
    c[0] = rfpart(yend) * xgap
    x[1] = xpxl1
    y[1] = ypxl1 + 1
    c[1] = fpart(yend) * xgap
    intery = yend + gradient  # first y-intersection for the main loop

    # handle second endpoint
    xend = np.round(x2)
    yend = y2 + gradient * (xend - x2)
    xgap = fpart(x2 + 0.5)
    xpxl2 = xend  # this will be used in the main loop
    ypxl2 = ipart(yend)
    x[2] = xpxl2
    y[2] = ypxl2
    c[2] = rfpart(yend) * xgap
    x[3] = xpxl2
    y[3] = ypxl2 + 1
    c[3] = fpart(yend) * xgap

    # main loop
    k = 4
    for i in range(int(xpxl1 + 1), int(xpxl2)):
        x[k] = i
        y[k] = ipart(intery)
        c[k] = rfpart(intery)
        k += 1
        x[k] = i
        y[k] = ipart(intery) + 1
        c[k] = fpart(intery)
        intery += gradient
        k += 1

    # truncate the vectors to proper sizes
    x = x[:k]
    y = y[:k]
    c = c[:k]

    if swapped:
        x, y = swap(x, y)

    return x, y, c

# integer part


def ipart(x):
    if x > 0:
        return int(np.floor(x))
    else:
        return int(np.ceil(x))

# round


def round(x):
    return ipart(x + 0.5)

# fractional part


def fpart(x):
    return x - ipart(x)

# RFractional part


def rfpart(x):
    return 1 - fpart(x)

# swap values


def swap(x, y):
    return y, x
