import numpy as np
import skimage.draw as draw
from string_art.transformations.xiaolinwu import xiaolinwu
from string_art.entities import Line, String

performance_mode = False


def draw_line(line: Line) -> String:
    if performance_mode:
        start, end = line
        x, y = draw.line(start[0], start[1], end[0], end[1])
        return x, y, np.ones_like(x)
    return xiaolinwu(line)
