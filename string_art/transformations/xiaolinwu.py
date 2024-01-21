import numpy as np
from string_art.entities import Line, String


def xiaolinwu(line: Line) -> String:
    start, end = line
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1

    # Preallocate memory for x, y, and c
    length = int(2.1 * np.sqrt(dx**2 + dy**2))
    x = np.zeros(length)
    y = np.zeros(length)
    c = np.zeros(length)

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

    # Handle first endpoint
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = rfpart(x1 + 0.5)
    xpxl1 = int(xend)  # this will be used in the main loop
    ypxl1 = int(yend)
    x[0] = xpxl1
    y[0] = ypxl1
    c[0] = rfpart(yend) * xgap
    x[1] = xpxl1
    y[1] = ypxl1 + 1
    c[1] = fpart(yend) * xgap
    intery = yend + gradient  # first y-intersection for the main loop

    # Handle second endpoint
    xend = round(x2)
    yend = y2 + gradient * (xend - x2)
    xgap = fpart(x2 + 0.5)
    xpxl2 = int(xend)  # this will be used in the main loop
    ypxl2 = int(yend)
    x[2] = xpxl2
    y[2] = ypxl2
    c[2] = rfpart(yend) * xgap
    x[3] = xpxl2
    y[3] = ypxl2 + 1
    c[3] = fpart(yend) * xgap

    # Main loop
    k = 4
    for i in range(xpxl1 + 1, xpxl2):
        x[k] = i
        y[k] = int(intery)
        c[k] = rfpart(intery)
        k += 1
        x[k] = i
        y[k] = int(intery) + 1
        c[k] = fpart(intery)
        intery = intery + gradient
        k += 1

    # Truncate the vectors to proper sizes
    x = x[:k]
    y = y[:k]
    c = c[:k]

    if swapped:
        x, y = swap(x, y)

    return x.astype(np.int32), y.astype(np.int32), c

# Integer part


def ipart(x):
    if x > 0:
        return int(np.floor(x))
    else:
        return int(np.ceil(x))

# Round function


def round(x):
    return ipart(x + 0.5)

# Fractional part


def fpart(x):
    return x - ipart(x)

# RF part


def rfpart(x):
    return 1 - fpart(x)

# Swap function


def swap(x, y):
    return y, x
