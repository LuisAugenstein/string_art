from string_art.entities import Line, String
import numpy as np


def lines_to_strings_in_positive_domain(lines: list[Line], domain_width: float) -> list[String]:
    strings: list[String] = []
    for line in lines:
        if line is None:
            strings.append(None)
        line = Line(np.round(line.end_points + 0.5 * domain_width))
        string = line.to_string()
        strings.append(string)
    return strings
