import numpy as np
from string_art.entities import Line, String


def xiaolinwu(line: Line) -> String:
    start, end = line

    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1

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
    xend, xend2 = round(x1), round(x2)
    yend, yend2 = y1 + gradient * (xend - x1), y2 + gradient * (xend2 - x2)
    xgap, xgap2 = rfpart(x1 + 0.5), fpart(x2 + 0.5)
    ypxl1, ypxl2 = np.trunc(yend), np.trunc(yend2)

    x = [xend, xend, xend2, xend2]
    y = [ypxl1, ypxl1+1, ypxl2, ypxl2+1]
    c = [rfpart(yend) * xgap, fpart(yend) * xgap, rfpart(yend) * xgap2, fpart(yend) * xgap2]

    i = np.arange(xend + 1, xend2)
    x = np.concatenate([x, i.repeat(2)])
    intery = yend + 1 + gradient + np.arange(i.shape[0])*gradient
    y = np.concatenate([y, np.array([ipart(intery)-1, ipart(intery)]).T.reshape(-1)])
    c = np.concatenate([c, np.array([rfpart(intery), fpart(intery)]).T.reshape(-1)])

    if swapped:
        x, y = swap(x, y)

    return x.astype(np.int32), y.astype(np.int32), c


def ipart(x: np.ndarray):
    if isinstance(x, np.ndarray):
        mask = x > 0
        x = x.copy()
        x[mask] = np.floor(x[mask])
        x[~mask] = np.ceil(x[~mask])
        return x

    if x > 0:
        return np.floor(x).astype(np.int32)
    else:
        return np.ceil(x).astype(np.int32)


def round(x):
    return ipart(x + 0.5)


def fpart(x):
    return x - ipart(x)


def rfpart(x):
    return 1 - fpart(x)


def swap(x, y):
    return y, x
