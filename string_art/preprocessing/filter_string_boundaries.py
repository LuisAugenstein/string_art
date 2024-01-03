from string_art.entities import String


def filter_string_boundaries(strings: list[String], resolution: int) -> list[String]:
    filtered_strings = []
    for string in strings:
        x, y, v = string.T
        mask = (x >= 0) & (x < resolution) & (y >= 0) & (y < resolution) & (v > 0) & (v <= 1)
        filtered_strings.append(string[mask])
    return filtered_strings
