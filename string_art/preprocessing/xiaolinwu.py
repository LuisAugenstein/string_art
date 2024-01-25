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

    x = np.array([x1, x2])
    y = np.array([y1, y2])
    xend = ipart(x + 0.5)
    yend = y + gradient * (xend - x)
    xgap = rfpart(x + 0.5)
    ypxl = np.trunc(yend)

    x = xend.repeat(2)
    y = [ypxl[0], ypxl[0]+1,
         ypxl[1], ypxl[1]+1]
    c = [rfpart(yend[0]) * xgap[0], fpart(yend[0]) * xgap[0],
         rfpart(yend[1]) * xgap[1], fpart(yend[1]) * xgap[1]]

    i = np.arange(xend[0] + 1, xend[1])
    x = np.concatenate([x, i.repeat(2)])
    intery = np.cumsum(np.concatenate([[yend[0] + 1 + gradient], np.ones(i.shape[0]-1)*gradient]))
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


def fpart(x):
    return x - ipart(x)


def rfpart(x):
    return 1 - fpart(x)


def swap(x, y):
    return y, x
