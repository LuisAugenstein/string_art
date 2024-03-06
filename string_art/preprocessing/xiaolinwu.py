import numpy as np
from string_art.entities import Line, String
import torch


def xiaolinwu(line: Line) -> String:
    """
    Parameters
    -
    line: torch.shape([2,2])
    """
    start, end = line

    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1

    swapped = False
    if torch.abs(dx) < torch.abs(dy):
        x1, y1 = swap(x1, y1)
        x2, y2 = swap(x2, y2)
        dx, dy = swap(dx, dy)
        swapped = True

    if x2 < x1:
        x1, x2 = swap(x1, x2)
        y1, y2 = swap(y1, y2)

    gradient = dy / dx

    x = torch.Tensor([x1, x2])
    y = torch.Tensor([y1, y2])
    xend = ipart(x + 0.5)
    yend = y + gradient * (xend - x)
    xgap = rfpart(x + 0.5)
    ypxl = torch.trunc(yend)

    x = xend.repeat_interleave(2)
    y = torch.stack([ypxl[0], ypxl[0]+1, ypxl[1], ypxl[1]+1])
    c = torch.stack([rfpart(yend)*xgap, fpart(yend)*xgap]).T.flatten()

    i = torch.arange(xend[0] + 1, xend[1])
    x = torch.cat([x, i.repeat_interleave(2)])
    catted = torch.cat([(yend[0] + 1 + gradient).unsqueeze(0), torch.ones(i.shape[0]-1)*gradient])
    intery = torch.cumsum(catted, dim=0)
    y = torch.cat([y, torch.stack([ipart(intery)-1, ipart(intery)]).T.reshape(-1)])
    c = torch.cat([c, torch.stack([rfpart(intery), fpart(intery)]).T.reshape(-1)])

    if swapped:
        x, y = swap(x, y)

    return x.to(torch.int32), y.to(torch.int32), c


def ipart(x: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    x: torch.shape([N])
    """
    mask = x > 0
    x = x.clone()
    x[mask] = np.floor(x[mask])
    x[~mask] = np.ceil(x[~mask])
    return x


def fpart(x):
    return x - ipart(x)


def rfpart(x):
    return 1 - fpart(x)


def swap(x, y):
    return y, x
